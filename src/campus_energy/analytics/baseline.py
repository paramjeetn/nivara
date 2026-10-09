"""Expected consumption model (baseline)."""
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class ExpectedConsumptionModel:
    """Predicts expected power consumption for an asset."""
    def __init__(self):
        # We use a simple linear regression with one-hot encoding for hour and day
        # and continuous for temperature and occupancy
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
            ("regressor", Ridge(alpha=1.0))
        ])
        self._is_fitted = False
        
    def fit(self, df: pd.DataFrame):
        """Train the model on normal data."""
        # Only use data where power is known
        valid = df.dropna(subset=["power_kw", "outdoor_temperature_c", "occupancy"])
        if len(valid) < 5:
            return
            
        X = valid[["hour", "day_of_week", "outdoor_temperature_c", "occupancy"]]
        y = valid["power_kw"]
        
        self.model.fit(X, y)
        self._is_fitted = True
        
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict expected power."""
        if not self._is_fitted:
            return pd.Series([0.0] * len(df), index=df.index)
            
        # Fill missing numeric with median or 0 to allow prediction
        X = df[["hour", "day_of_week", "outdoor_temperature_c", "occupancy"]].copy()
        X["outdoor_temperature_c"] = X["outdoor_temperature_c"].fillna(25.0)
        X["occupancy"] = X["occupancy"].fillna(0)
        
        return pd.Series(self.model.predict(X), index=df.index)
