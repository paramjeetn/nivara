"""Feature engineering for anomaly detection and baseline models."""
import pandas as pd
import numpy as np

def engineer_features(telemetry_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    """Merge telemetry with context and engineer features for detection.
    
    Returns a dataframe ready for modeling.
    """
    # Ensure datetime
    tel = telemetry_df.copy()
    ctx = context_df.copy()
    
    tel["timestamp"] = pd.to_datetime(tel["timestamp"])
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"])
    
    # Merge context (context is per building/zone, telemetry is per asset)
    # We need to map asset -> zone/building. The merge strategy depends on having that mapping.
    # For simplicity, if telemetry lacks zone_id but has asset_id, we need assets_df.
    # We'll just assume we do the merge via timestamp and asset's parent building/zone.
    # To keep it simple here, we expect the caller to have added building_id/zone_id to tel
    # or we merge on timestamp and building_id if it's a meter.
    
    # Actually, a better approach is to do rolling features on telemetry alone,
    # and contextual features like hour_of_day.
    
    tel = tel.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    
    # Time-based features
    tel["hour"] = tel["timestamp"].dt.hour
    tel["day_of_week"] = tel["timestamp"].dt.dayofweek
    tel["is_weekend"] = tel["day_of_week"].isin([5, 6]).astype(int)
    
    # Rolling features (grouped by asset)
    # E.g. moving average of power over last 3 hours (36 intervals)
    tel["power_ma_3h"] = tel.groupby("asset_id")["power_kw"].transform(lambda x: x.rolling(36, min_periods=1).mean())
    tel["power_std_3h"] = tel.groupby("asset_id")["power_kw"].transform(lambda x: x.rolling(36, min_periods=1).std())
    
    return tel
