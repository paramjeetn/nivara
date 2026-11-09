"""Expected consumption model (baseline) with empirical prediction intervals."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class ExpectedConsumptionModel:
    """Predicts expected power consumption for an asset.

    Degrades gracefully when context columns (outdoor_temperature_c, occupancy)
    are absent from the features dataframe — fills safe defaults instead of crashing.
    Computes empirical residual quantiles to yield calibrated prediction intervals.
    """

    def __init__(self):
        categorical_features = ["hour", "day_of_week"]
        numeric_features = ["outdoor_temperature_c", "occupancy"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("num", "passthrough", numeric_features),
            ]
        )

        self.model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ])
        self._is_fitted = False
        self.residuals_: np.ndarray = np.array([])
        self.lower_residual_offset_: float = 0.0
        self.upper_residual_offset_: float = 0.0

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required columns exist with safe defaults."""
        out = df.copy()
        if "outdoor_temperature_c" not in out.columns:
            out["outdoor_temperature_c"] = 32.0   # Odisha annual mean °C
        if "occupancy" not in out.columns:
            out["occupancy"] = 0
        out["outdoor_temperature_c"] = out["outdoor_temperature_c"].fillna(32.0)
        out["occupancy"] = out["occupancy"].fillna(0)
        return out

    def fit(
        self,
        df: pd.DataFrame,
        lower_quantile: float = 10.0,
        upper_quantile: float = 90.0,
    ):
        """Train the model on normal (non-anomalous) data and compute residual quantiles."""
        df = self._prepare(df)
        valid = df.dropna(subset=["power_kw"])
        if len(valid) < 5:
            return

        X = valid[["hour", "day_of_week", "outdoor_temperature_c", "occupancy"]]
        y = valid["power_kw"]
        self.model.fit(X, y)
        self._is_fitted = True

        # Compute empirical residuals and prediction interval offsets
        train_preds = self.model.predict(X)
        residuals = (y.values - train_preds).astype(float)
        self.residuals_ = residuals
        if len(residuals) > 0:
            self.lower_residual_offset_ = float(np.percentile(residuals, lower_quantile))
            self.upper_residual_offset_ = float(np.percentile(residuals, upper_quantile))

    def predict_interval(
        self,
        df: pd.DataFrame,
        lower_quantile: float = 10.0,
        upper_quantile: float = 90.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Predict expected power consumption along with empirical prediction intervals.

        Computes [yhat_lower, yhat_upper] based on empirical residual quantiles.
        Guarantees: 0 <= yhat_lower <= yhat <= yhat_upper.

        Returns:
            tuple of (yhat, yhat_lower, yhat_upper) as pandas Series.
        """
        if not self._is_fitted:
            zero_series = pd.Series([0.0] * len(df), index=df.index)
            return zero_series, zero_series, zero_series

        df_prep = self._prepare(df)
        X = df_prep[["hour", "day_of_week", "outdoor_temperature_c", "occupancy"]].copy()
        raw_preds = self.model.predict(X)
        yhat = pd.Series(np.clip(raw_preds, 0.0, None), index=df.index)

        # Quantile offsets
        if hasattr(self, "residuals_") and len(self.residuals_) >= 5:
            q_low = float(np.percentile(self.residuals_, lower_quantile))
            q_high = float(np.percentile(self.residuals_, upper_quantile))
        else:
            q_low = getattr(self, "lower_residual_offset_", 0.0)
            q_high = getattr(self, "upper_residual_offset_", 0.0)

        if q_high <= q_low:
            std_res = float(np.std(self.residuals_)) if len(self.residuals_) else 1.0
            q_low = -1.28 * max(std_res, 0.5)
            q_high = 1.28 * max(std_res, 0.5)

        yhat_lower = pd.Series(np.clip(raw_preds + q_low, 0.0, None), index=df.index)
        yhat_upper = pd.Series(np.clip(raw_preds + q_high, 0.0, None), index=df.index)

        # Strictly enforce bounds: 0 <= yhat_lower <= yhat <= yhat_upper
        yhat_lower = pd.Series(
            np.clip(np.minimum(yhat_lower.values, yhat.values), 0.0, None),
            index=df.index,
        )
        yhat_upper = pd.Series(
            np.maximum(yhat_upper.values, yhat.values),
            index=df.index,
        )

        return yhat, yhat_lower, yhat_upper

    def predict(
        self,
        df: pd.DataFrame,
        return_intervals: bool = False,
        lower_quantile: float = 10.0,
        upper_quantile: float = 90.0,
    ) -> pd.Series | tuple[pd.Series, pd.Series, pd.Series]:
        """Predict expected power for each row.

        If return_intervals is True, returns (yhat, yhat_lower, yhat_upper).
        Otherwise returns yhat.
        """
        if return_intervals:
            return self.predict_interval(
                df, lower_quantile=lower_quantile, upper_quantile=upper_quantile
            )

        if not self._is_fitted:
            return pd.Series([0.0] * len(df), index=df.index)

        df_prep = self._prepare(df)
        X = df_prep[["hour", "day_of_week", "outdoor_temperature_c", "occupancy"]].copy()
        return pd.Series(self.model.predict(X), index=df.index)

