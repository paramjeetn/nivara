"""Forecasting module to project energy consumption 24-hours forward."""
import pandas as pd
from datetime import timedelta
import numpy as np
from campus_energy.analytics.baseline import ExpectedConsumptionModel
from campus_energy.analytics.features import engineer_features

def generate_24h_forecast(telemetry_df: pd.DataFrame, context_df: pd.DataFrame, assets_df: pd.DataFrame, temp_offset: float = 0.0) -> pd.DataFrame:
    """Predict the next 24 hours of campus energy load.
    
    In a real scenario, this would use Prophet and a weather forecast API.
    For this prototype, we simulate a 'future context' and use our
    ExpectedConsumptionModel (Ridge regression) to project the load.
    """
    tel = telemetry_df.copy()
    tel["timestamp"] = pd.to_datetime(tel["timestamp"])
    if tel["timestamp"].dt.tz is not None:
        tel["timestamp"] = tel["timestamp"].dt.tz_localize(None)
    
    # Get the latest timestamp
    latest_ts = tel["timestamp"].max()
    
    # We will build a forecast only for building main meters to keep it clean
    meters = assets_df[assets_df["asset_type"] == "meter"]["asset_id"].tolist()
    meter_tel = tel[tel["asset_id"].isin(meters)]
    
    # We train the model on the last 14 days of data
    train_start = latest_ts - timedelta(days=14)
    train_data = meter_tel[meter_tel["timestamp"] >= train_start].copy()
    
    # Engineer basic features for training
    train_data["hour"] = train_data["timestamp"].dt.hour
    train_data["day_of_week"] = train_data["timestamp"].dt.dayofweek
    train_data["occupancy"] = 0 # Meters don't have occupancy, they aggregate. We just use time/weather.
    
    # We need weather for meters. We approximate by taking the mean weather of the campus at each TS
    ctx_weather = context_df.copy()
    ctx_weather["timestamp"] = pd.to_datetime(ctx_weather["timestamp"])
    if ctx_weather["timestamp"].dt.tz is not None:
        ctx_weather["timestamp"] = ctx_weather["timestamp"].dt.tz_localize(None)
        
    campus_weather = ctx_weather.groupby("timestamp")[["outdoor_temperature_c"]].mean().reset_index()
    
    train_data = train_data.merge(campus_weather, on="timestamp", how="left")
    
    forecasts = []
    
    # Predict 24 hours forward (5-min intervals = 288 steps)
    future_timestamps = [latest_ts + timedelta(minutes=5 * i) for i in range(1, 289)]
    
    for meter_id in meters:
        asset_train = train_data[train_data["asset_id"] == meter_id]
        if len(asset_train) < 100:
            continue
            
        model = ExpectedConsumptionModel()
        model.fit(asset_train)
        
        # Build future dataframe
        future_df = pd.DataFrame({"timestamp": future_timestamps})
        future_df["hour"] = future_df["timestamp"].dt.hour
        future_df["day_of_week"] = future_df["timestamp"].dt.dayofweek
        future_df["asset_id"] = meter_id
        
        # Mock weather forecast (just repeat yesterday's weather + offset)
        yesterday_weather = campus_weather[(campus_weather["timestamp"] > latest_ts - timedelta(days=1)) & (campus_weather["timestamp"] <= latest_ts)]
        # ensure alignment
        if len(yesterday_weather) == 288:
            future_df["outdoor_temperature_c"] = yesterday_weather["outdoor_temperature_c"].values + temp_offset
        else:
            future_df["outdoor_temperature_c"] = 32.0 + temp_offset
            
        future_df["occupancy"] = 0
        
        # Predict
        preds = model.predict(future_df)
        
        # Add bounds and uncertainty
        future_df["predicted_power_kw"] = np.clip(preds, 0, None)
        future_df["lower_bound"] = np.clip(preds * 0.85, 0, None) # Simple 15% confidence band
        future_df["upper_bound"] = preds * 1.15
        
        forecasts.append(future_df[["timestamp", "asset_id", "predicted_power_kw", "lower_bound", "upper_bound"]])
        
    if not forecasts:
        return pd.DataFrame()
        
    return pd.concat(forecasts, ignore_index=True)
