"""Forecasting module: projects campus energy 24-hours forward.

Uses a Ridge regression baseline per asset (main meters only).
Features include: hour-of-day, day-of-week, is_holiday, outdoor temperature,
rolling 3h mean power, and rolling 3h std power.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from campus_energy.analytics.baseline import ExpectedConsumptionModel


def generate_24h_forecast(
    telemetry_df: pd.DataFrame,
    context_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    temp_offset: float = 0.0,
    ac_policy_reduction_pct: float = 0.0,
    occupancy_reduction_pct: float = 0.0,   # NEW: % reduction in occupancy (holiday/exam)
    solar_pv_kwp: float = 0.0,               # NEW: kWp of rooftop solar to subtract from grid import
    led_upgrade: bool = False,               # NEW: replace fluorescent with LED (40% lighting reduction)
) -> pd.DataFrame:
    """Predict next 24 hours of campus energy load.

    Args:
        telemetry_df: Full telemetry history.
        context_df: Context (weather + occupancy) history.
        assets_df: Asset registry.
        temp_offset: Temperature delta for weather scenario simulation (°C).
        ac_policy_reduction_pct: % reduction applied after 18:00 (AC policy scenario).
        occupancy_reduction_pct: % reduction in occupancy (holiday/exam).
        solar_pv_kwp: kWp of rooftop solar to subtract from grid import.
        led_upgrade: Replace fluorescent with LED (40% lighting reduction).

    Returns:
        DataFrame with columns:
            timestamp, asset_id, predicted_power_kw,
            lower_bound, upper_bound, scenario_power_kw
    """
    tel = telemetry_df.copy()
    tel["timestamp"] = pd.to_datetime(tel["timestamp"])
    if tel["timestamp"].dt.tz is not None:
        tel["timestamp"] = tel["timestamp"].dt.tz_localize(None)

    ctx = context_df.copy()
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"])
    if ctx["timestamp"].dt.tz is not None:
        ctx["timestamp"] = ctx["timestamp"].dt.tz_localize(None)

    latest_ts = tel["timestamp"].max()

    # Only forecast on building main meters (not sub-equipment)
    meters = assets_df[assets_df["asset_type"] == "meter"]["asset_id"].tolist()
    meter_tel = tel[tel["asset_id"].isin(meters)].copy()

    # Build campus-level weather context
    campus_weather = ctx.groupby("timestamp")[["outdoor_temperature_c"]].mean().reset_index()

    # Identify holidays from context
    if "holiday" in ctx.columns:
        holiday_dates = set(ctx[ctx["holiday"] == True]["timestamp"].dt.date.tolist())
    else:
        holiday_dates = set()

    # Train on last 21 days of data
    train_start = latest_ts - timedelta(days=21)
    train_base = meter_tel[meter_tel["timestamp"] >= train_start].copy()
    train_base = train_base.merge(campus_weather, on="timestamp", how="left")

    # Engineer features
    for df in [train_base]:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_holiday"] = df["timestamp"].dt.date.isin(holiday_dates).astype(int)
        df["outdoor_temperature_c"] = df["outdoor_temperature_c"].fillna(32.0)
        df["occupancy"] = 0  # Meters aggregate; use time features instead

        # Rolling features (per meter per training set)
        df["power_ma_3h"] = (
            df.groupby("asset_id")["power_kw"]
            .transform(lambda x: x.rolling(36, min_periods=1).mean())
        )
        df["power_std_3h"] = (
            df.groupby("asset_id")["power_kw"]
            .transform(lambda x: x.rolling(36, min_periods=1).std().fillna(0))
        )

    # Future timestamps: next 24 hours in 5-min intervals
    future_timestamps = [latest_ts + timedelta(minutes=5 * i) for i in range(1, 289)]

    # Yesterday's weather as a proxy for tomorrow (+ temp_offset)
    yesterday_weather = campus_weather[
        (campus_weather["timestamp"] > latest_ts - timedelta(days=1))
        & (campus_weather["timestamp"] <= latest_ts)
    ]["outdoor_temperature_c"].values

    forecasts = []

    for meter_id in meters:
        asset_train = train_base[train_base["asset_id"] == meter_id].dropna(subset=["power_kw"])
        if len(asset_train) < 100:
            continue

        # Fit an extended baseline model
        model = ExpectedConsumptionModel()
        model.fit(asset_train)

        # Build future feature frame
        future_df = pd.DataFrame({"timestamp": future_timestamps})
        future_df["hour"] = future_df["timestamp"].dt.hour
        future_df["day_of_week"] = future_df["timestamp"].dt.dayofweek
        future_df["is_weekend"] = future_df["day_of_week"].isin([5, 6]).astype(int)
        future_df["is_holiday"] = future_df["timestamp"].dt.date.isin(holiday_dates).astype(int)
        future_df["asset_id"] = meter_id
        future_df["occupancy"] = 0

        if len(yesterday_weather) == 288:
            future_df["outdoor_temperature_c"] = yesterday_weather + temp_offset
        else:
            future_df["outdoor_temperature_c"] = 32.0 + temp_offset

        # Carry forward rolling features from last known point
        last_ma = asset_train["power_ma_3h"].iloc[-1] if len(asset_train) else 0.0
        last_std = asset_train["power_std_3h"].iloc[-1] if len(asset_train) else 0.0
        future_df["power_ma_3h"] = last_ma
        future_df["power_std_3h"] = last_std

        preds = model.predict(future_df)
        future_df["predicted_power_kw"] = np.clip(preds, 0, None)

        # Uncertainty bands scale with std
        future_df["lower_bound"] = np.clip(future_df["predicted_power_kw"] * 0.88, 0, None)
        future_df["upper_bound"] = future_df["predicted_power_kw"] * 1.12

        scenario_power = future_df["predicted_power_kw"].copy()

        # Occupancy reduction: scale down predicted load proportionally
        # Assumption: 60% of building load is occupancy-driven
        if occupancy_reduction_pct > 0:
            occ_factor = 1 - (occupancy_reduction_pct / 100.0) * 0.60
            scenario_power = scenario_power * occ_factor

        # Solar PV: subtract average solar generation (assume 4 peak sun hours/day, 5-min intervals)
        # Solar generation per 5-min interval = solar_pv_kwp * (4/24) * (5/60) but clamp to daylight hours (6-18)
        if solar_pv_kwp > 0:
            is_daylight = future_df["hour"].between(6, 18)
            solar_per_interval_kw = solar_pv_kwp * 0.80  # 80% capacity factor during daylight
            scenario_power = scenario_power - (is_daylight * solar_per_interval_kw)
            scenario_power = scenario_power.clip(lower=0)

        # LED upgrade: reduce lighting load by 40% (lighting typically 20-25% of total load)
        if led_upgrade:
            lighting_fraction = 0.22  # 22% of total building load is lighting
            led_saving_factor = 0.40  # LED saves 40% vs fluorescent
            scenario_power = scenario_power * (1 - lighting_fraction * led_saving_factor)

        # Apply AC policy on top
        if ac_policy_reduction_pct > 0:
            after_6pm = future_df["hour"] >= 18
            scenario_power = scenario_power.copy()
            scenario_power[after_6pm] *= (1 - ac_policy_reduction_pct / 100.0)

        future_df["scenario_power_kw"] = scenario_power

        forecasts.append(
            future_df[
                [
                    "timestamp",
                    "asset_id",
                    "predicted_power_kw",
                    "lower_bound",
                    "upper_bound",
                    "scenario_power_kw",
                ]
            ]
        )

    if not forecasts:
        return pd.DataFrame()

    return pd.concat(forecasts, ignore_index=True)
