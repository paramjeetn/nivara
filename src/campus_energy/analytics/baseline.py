"""Expected consumption model (baseline)."""
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class ExpectedConsumptionModel:
    """Predicts expected power consumption for an asset.

    Degrades gracefully when context columns (outdoor_temperature_c, occupancy)
    are absent from the features dataframe — fills safe defaults instead of crashing.
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

    def fit(self, df: pd.DataFrame):
        """Train the model on normal (non-anomalous) data."""
        df = self._prepare(df)
        valid = df.dropna(subset=["power_kw"])
        if len(valid) < 5:
            return

        X = valid[["hour", "day_of_week", "outdoor_temperature_c", "occupancy"]]
        y = valid["power_kw"]
        self.model.fit(X, y)
        self._is_fitted = True

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict expected power for each row."""
        if not self._is_fitted:
            return pd.Series([0.0] * len(df), index=df.index)

        df = self._prepare(df)
        X = df[["hour", "day_of_week", "outdoor_temperature_c", "occupancy"]].copy()
        return pd.Series(self.model.predict(X), index=df.index)
