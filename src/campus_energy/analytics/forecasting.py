"""Forecasting module: projects campus energy 24-hours forward.

Uses a Ridge regression baseline per asset (main meters only).
Features include: hour-of-day, day-of-week, is_holiday, outdoor temperature,
rolling 3h mean power, and rolling 3h std power.
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Optional, Union, Sequence, Dict, Any, List
from sklearn.linear_model import Ridge
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

        # Fit an extended baseline model with empirical prediction intervals
        model = ExpectedConsumptionModel()
        model.fit(asset_train, lower_quantile=10.0, upper_quantile=90.0)

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

        # Predict baseline and empirical residual prediction intervals (80% interval)
        yhat, yhat_lower, yhat_upper = model.predict(
            future_df, return_intervals=True, lower_quantile=10.0, upper_quantile=90.0
        )
        future_df["predicted_power_kw"] = np.clip(yhat, 0, None)
        future_df["lower_bound"] = np.clip(yhat_lower, 0, None)
        future_df["upper_bound"] = np.clip(yhat_upper, 0, None)
        future_df["baseline_lower"] = future_df["lower_bound"]
        future_df["baseline_upper"] = future_df["upper_bound"]

        scenario_power = future_df["predicted_power_kw"].copy()
        scenario_lower = future_df["lower_bound"].copy()
        scenario_upper = future_df["upper_bound"].copy()

        # Occupancy reduction: scale down predicted load proportionally
        # Assumption: 60% of building load is occupancy-driven
        if occupancy_reduction_pct > 0:
            occ_factor = 1.0 - (occupancy_reduction_pct / 100.0) * 0.60
            scenario_power = scenario_power * occ_factor
            scenario_lower = scenario_lower * occ_factor
            scenario_upper = scenario_upper * occ_factor

        # Solar PV: subtract average solar generation (assume 4 peak sun hours/day, 5-min intervals)
        if solar_pv_kwp > 0:
            is_daylight = future_df["hour"].between(6, 18).astype(float)
            solar_per_interval_kw = solar_pv_kwp * 0.80  # 80% capacity factor during daylight
            solar_reduction = is_daylight * solar_per_interval_kw
            scenario_power = np.clip(scenario_power - solar_reduction, 0, None)
            scenario_lower = np.clip(scenario_lower - solar_reduction, 0, None)
            scenario_upper = np.clip(scenario_upper - solar_reduction, 0, None)

        # LED upgrade: reduce lighting load by 40% (lighting typically 20-25% of total load)
        if led_upgrade:
            lighting_fraction = 0.22  # 22% of total building load is lighting
            led_saving_factor = 0.40  # LED saves 40% vs fluorescent
            led_factor = 1.0 - lighting_fraction * led_saving_factor
            scenario_power = scenario_power * led_factor
            scenario_lower = scenario_lower * led_factor
            scenario_upper = scenario_upper * led_factor

        # Apply AC policy on top
        if ac_policy_reduction_pct > 0:
            after_6pm = (future_df["hour"] >= 18).astype(float)
            ac_factor = 1.0 - (ac_policy_reduction_pct / 100.0) * after_6pm
            scenario_power = scenario_power * ac_factor
            scenario_lower = scenario_lower * ac_factor
            scenario_upper = scenario_upper * ac_factor

        scenario_lower = np.clip(np.minimum(scenario_lower, scenario_power), 0, None)
        scenario_upper = np.maximum(scenario_upper, scenario_power)

        future_df["scenario_power_kw"] = scenario_power
        future_df["scenario_lower"] = scenario_lower
        future_df["scenario_upper"] = scenario_upper

        forecasts.append(
            future_df[
                [
                    "timestamp",
                    "asset_id",
                    "predicted_power_kw",
                    "lower_bound",
                    "upper_bound",
                    "baseline_lower",
                    "baseline_upper",
                    "scenario_power_kw",
                    "scenario_lower",
                    "scenario_upper",
                ]
            ]
        )

    if not forecasts:
        return pd.DataFrame()

    return pd.concat(forecasts, ignore_index=True)


def compute_forecast_savings_range(
    forecast_df: pd.DataFrame,
    cost_per_kwh: float | None = None,
    grid_emission_factor: float | None = None,
) -> dict[str, dict[str, float]]:
    """Compute 24-hour scenario savings ranges {low, expected, high}.

    Parameters:
        forecast_df: Output from generate_24h_forecast.
        cost_per_kwh: Commercial electricity tariff in INR/kWh (defaults to assumptions registry).
        grid_emission_factor: Grid emission factor in kgCO2e/kWh (defaults to assumptions registry).

    Returns:
        dict with keys 'kwh', 'cost_inr', 'co2_kg', each having subkeys 'low', 'expected', 'high'.
    """
    if forecast_df.empty or "scenario_power_kw" not in forecast_df.columns:
        empty_range = {"low": 0.0, "expected": 0.0, "high": 0.0}
        return {"kwh": empty_range, "cost_inr": empty_range, "co2_kg": empty_range}

    if cost_per_kwh is None:
        try:
            from campus_energy import assumptions
            cost_per_kwh = float(assumptions.get_value("tariff.electricity_commercial"))
        except Exception:
            cost_per_kwh = 7.50

    if grid_emission_factor is None:
        try:
            from campus_energy import assumptions
            grid_emission_factor = float(assumptions.get_value("emissions.grid_factor"))
        except Exception:
            grid_emission_factor = 0.82

    # Energy reduction across 5-minute intervals (12 intervals / hour)
    saved_kw_expected = np.clip(
        forecast_df["predicted_power_kw"] - forecast_df["scenario_power_kw"], 0, None
    )
    kwh_expected = float(saved_kw_expected.sum() / 12.0)

    # Uncertainty bounds on savings
    base_low = (
        forecast_df["baseline_lower"]
        if "baseline_lower" in forecast_df.columns
        else forecast_df["lower_bound"]
    )
    scen_low = (
        forecast_df["scenario_lower"]
        if "scenario_lower" in forecast_df.columns
        else forecast_df["scenario_power_kw"]
    )
    base_high = (
        forecast_df["baseline_upper"]
        if "baseline_upper" in forecast_df.columns
        else forecast_df["upper_bound"]
    )
    scen_high = (
        forecast_df["scenario_upper"]
        if "scenario_upper" in forecast_df.columns
        else forecast_df["scenario_power_kw"]
    )

    saved_kw_low = np.clip(base_low - scen_low, 0, None)
    kwh_low = float(saved_kw_low.sum() / 12.0)

    saved_kw_high = np.clip(base_high - scen_high, 0, None)
    kwh_high = float(saved_kw_high.sum() / 12.0)

    # Monotonicity check
    kwh_low = min(kwh_low, kwh_expected)
    kwh_high = max(kwh_high, kwh_expected)

    cost_low = round(kwh_low * cost_per_kwh, 2)
    cost_exp = round(kwh_expected * cost_per_kwh, 2)
    cost_high = round(kwh_high * cost_per_kwh, 2)

    co2_low = round(kwh_low * grid_emission_factor, 2)
    co2_exp = round(kwh_expected * grid_emission_factor, 2)
    co2_high = round(kwh_high * grid_emission_factor, 2)

    return {
        "kwh": {"low": round(kwh_low, 1), "expected": round(kwh_expected, 1), "high": round(kwh_high, 1)},
        "cost_inr": {"low": cost_low, "expected": cost_exp, "high": cost_high},
        "co2_kg": {"low": co2_low, "expected": co2_exp, "high": co2_high},
    }


def generate_production_forecast(
    telemetry_df: pd.DataFrame,
    context_df: Optional[pd.DataFrame] = None,
    assets_df: Optional[pd.DataFrame] = None,
    horizon_hours: int = 24,
    temp_offset: float = 0.0,
    ac_policy_reduction_pct: float = 0.0,
    occupancy_reduction_pct: float = 0.0,
    solar_pv_kwp: float = 0.0,
    led_upgrade: bool = False,
    return_campus_total: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Generate production 24-hour hybrid forecast for operational campus energy planning.

    Architecture (Redesign Spec Section 11):
        1. Seasonal-Naive Baseline: Anchors to exact same time slot 7 days prior (t - 7d).
           If missing, gracefully falls back to yesterday (t - 1d), then intraday slot mean.
        2. Ridge Residual ML Model: Predicts diurnal, weather (delta-T), occupancy, and holiday
           residual variance on top of the seasonal calendar baseline.
        3. Calibrated 80% Empirical Prediction Intervals: Computes empirical residual quantiles
           (10th & 90th percentiles) guaranteeing 0 <= lower_bound <= predicted <= upper_bound.
        4. Planning Levers: Supports what-if scenario parameter shifts (temp offset, AC policy,
           occupancy reduction, rooftop solar PV, LED retrofit).

    Args:
        telemetry_df: Historical telemetry records (timestamp, power_kw, optional asset_id).
        context_df: Historical environmental context (temperature, occupancy, holiday).
        assets_df: Asset registry dataframe.
        horizon_hours: Forward forecast horizon in hours (default 24h = 288 x 5-min intervals).
        temp_offset: Simulated ambient temperature deviation in °C.
        ac_policy_reduction_pct: Policy cutback percentage applied after 18:00.
        occupancy_reduction_pct: Percentage reduction in campus occupancy.
        solar_pv_kwp: Additional rooftop solar PV rating to subtract from grid import.
        led_upgrade: Whether to simulate 40% reduction on lighting baseload.
        return_campus_total: If True, aggregates to a single campus total series.

    Returns:
        pd.DataFrame containing:
            timestamp, predicted_power_kw, lower_bound, upper_bound,
            baseline_power_kw, scenario_power_kw, scenario_lower, scenario_upper,
            (and asset_id if per-meter forecast).
    """
    empty_cols = [
        "timestamp",
        "predicted_power_kw",
        "lower_bound",
        "upper_bound",
        "baseline_power_kw",
        "scenario_power_kw",
        "scenario_lower",
        "scenario_upper",
    ]
    if telemetry_df is None or telemetry_df.empty:
        return pd.DataFrame(columns=empty_cols)

    # Defensive argument swapping
    if isinstance(assets_df, (int, float)):
        horizon_hours = int(assets_df)
        assets_df = None
    if isinstance(context_df, pd.DataFrame) and "asset_type" in context_df.columns and assets_df is None:
        assets_df = context_df
        context_df = None

    tel = telemetry_df.copy()
    tel["timestamp"] = pd.to_datetime(tel["timestamp"])
    if tel["timestamp"].dt.tz is not None:
        tel["timestamp"] = tel["timestamp"].dt.tz_localize(None)

    # Context prep
    ctx_weather_map: Dict[pd.Timestamp, float] = {}
    ctx_occ_map: Dict[pd.Timestamp, float] = {}
    holiday_dates = set()

    if context_df is not None and not context_df.empty:
        ctx = context_df.copy()
        ctx["timestamp"] = pd.to_datetime(ctx["timestamp"])
        if ctx["timestamp"].dt.tz is not None:
            ctx["timestamp"] = ctx["timestamp"].dt.tz_localize(None)

        if "outdoor_temperature_c" in ctx.columns:
            w_mean = ctx.groupby("timestamp")["outdoor_temperature_c"].mean()
            ctx_weather_map = w_mean.to_dict()

        if "occupancy" in ctx.columns:
            o_mean = ctx.groupby("timestamp")["occupancy"].mean()
            ctx_occ_map = o_mean.to_dict()

        if "holiday" in ctx.columns:
            holiday_dates = set(ctx[ctx["holiday"] == True]["timestamp"].dt.date.tolist())

    latest_ts = tel["timestamp"].max()
    num_intervals = int(horizon_hours * 12)
    future_timestamps = [latest_ts + timedelta(minutes=5 * i) for i in range(1, num_intervals + 1)]

    # Determine meter groups
    meters_to_forecast: List[Optional[str]] = []
    if assets_df is not None and "asset_type" in assets_df.columns:
        m_list = assets_df[assets_df["asset_type"] == "meter"]["asset_id"].tolist()
        if m_list:
            meters_to_forecast = m_list

    if not meters_to_forecast:
        # Default: single aggregate campus load forecast
        meters_to_forecast = [None]

    # If return_campus_total requested and multiple meters exist, we aggregate across meters
    # Or if single meter / None, we forecast single series
    series_list = []

    for m_id in meters_to_forecast:
        if m_id is not None and "asset_id" in tel.columns:
            sub_tel = tel[tel["asset_id"] == m_id].copy()
        else:
            sub_tel = tel.copy()

        if sub_tel.empty:
            continue

        # Aggregate 5-minute series for this meter or campus
        m_series = (
            sub_tel.groupby("timestamp")["power_kw"]
            .sum()
            .reset_index()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(m_series) < 10:
            continue

        # 1. Seasonal-Naive Baseline Map
        history_map = dict(zip(m_series["timestamp"], m_series["power_kw"].astype(float)))
        m_series["_hr"] = m_series["timestamp"].dt.hour
        m_series["_min"] = m_series["timestamp"].dt.minute
        slot_means = m_series.groupby(["_hr", "_min"])["power_kw"].mean().to_dict()
        global_mean = float(m_series["power_kw"].mean()) if len(m_series) else 0.0

        # Generate future seasonal-naive predictions
        sn_future = []
        for fts in future_timestamps:
            t_week = fts - timedelta(days=7)
            if t_week in history_map:
                sn_future.append(history_map[t_week])
            elif (fts - timedelta(days=1)) in history_map:
                sn_future.append(history_map[fts - timedelta(days=1)])
            elif (fts.hour, fts.minute) in slot_means:
                sn_future.append(slot_means[(fts.hour, fts.minute)])
            else:
                sn_future.append(global_mean)
        sn_future = np.clip(np.array(sn_future, dtype=float), 0.0, None)

        # 2. Residual Model Training (Past 21 Days)
        train_start = latest_ts - timedelta(days=21)
        train_sub = m_series[m_series["timestamp"] >= train_start].copy()

        # Compute historical seasonal-naive baseline on training set
        train_sn = [
            history_map.get(
                t - timedelta(days=7),
                history_map.get(t - timedelta(days=1), slot_means.get((t.hour, t.minute), global_mean)),
            )
            for t in train_sub["timestamp"]
        ]
        train_sub["sn_baseline"] = train_sn
        train_sub["residual"] = train_sub["power_kw"] - train_sub["sn_baseline"]

        # Features for residual Ridge model
        train_sub["temp_now"] = [ctx_weather_map.get(t, 32.0) for t in train_sub["timestamp"]]
        train_sub["temp_7d"] = [ctx_weather_map.get(t - timedelta(days=7), 32.0) for t in train_sub["timestamp"]]
        train_sub["delta_temp"] = train_sub["temp_now"] - train_sub["temp_7d"]

        train_sub["occ_now"] = [ctx_occ_map.get(t, 0.0) for t in train_sub["timestamp"]]
        train_sub["occ_7d"] = [ctx_occ_map.get(t - timedelta(days=7), 0.0) for t in train_sub["timestamp"]]
        train_sub["delta_occ"] = train_sub["occ_now"] - train_sub["occ_7d"]

        train_sub["is_holiday_now"] = train_sub["timestamp"].dt.date.isin(holiday_dates).astype(int)
        train_sub["is_holiday_7d"] = (train_sub["timestamp"] - timedelta(days=7)).dt.date.isin(holiday_dates).astype(int)
        train_sub["delta_holiday"] = train_sub["is_holiday_now"] - train_sub["is_holiday_7d"]

        feat_cols = ["delta_temp", "delta_occ", "delta_holiday"]
        X_train = train_sub[feat_cols].fillna(0.0)
        y_train = train_sub["residual"].fillna(0.0)

        ridge = Ridge(alpha=100.0)
        fitted_ok = False
        empirical_q10 = -5.0
        empirical_q90 = 5.0

        if len(train_sub) >= 50:
            try:
                ridge.fit(X_train, y_train)
                fitted_ok = True
                fitted_residuals = y_train.values - ridge.predict(X_train)
                empirical_q10 = float(np.percentile(fitted_residuals, 10))
                empirical_q90 = float(np.percentile(fitted_residuals, 90))
            except Exception:
                fitted_ok = False

        if empirical_q90 <= empirical_q10:
            std_r = float(np.std(y_train)) if len(y_train) else 5.0
            empirical_q10 = -1.28 * max(std_r, 1.0)
            empirical_q90 = 1.28 * max(std_r, 1.0)

        # 3. Predict Residual on Future Timestamps
        future_df = pd.DataFrame({"timestamp": future_timestamps})
        future_df["temp_now"] = [ctx_weather_map.get(t, 32.0) + temp_offset for t in future_timestamps]
        future_df["temp_7d"] = [ctx_weather_map.get(t - timedelta(days=7), 32.0) for t in future_timestamps]
        future_df["delta_temp"] = future_df["temp_now"] - future_df["temp_7d"]

        future_df["occ_now"] = [ctx_occ_map.get(t, 0.0) for t in future_timestamps]
        future_df["occ_7d"] = [ctx_occ_map.get(t - timedelta(days=7), 0.0) for t in future_timestamps]
        future_df["delta_occ"] = future_df["occ_now"] - future_df["occ_7d"]

        future_df["is_holiday_now"] = future_df["timestamp"].dt.date.isin(holiday_dates).astype(int)
        future_df["is_holiday_7d"] = (future_df["timestamp"] - timedelta(days=7)).dt.date.isin(holiday_dates).astype(int)
        future_df["delta_holiday"] = future_df["is_holiday_now"] - future_df["is_holiday_7d"]

        if fitted_ok:
            pred_residuals = ridge.predict(future_df[feat_cols].fillna(0.0))
            # Safe clipping of residual to prevent excessive runaway deviation (> 25% of baseline)
            max_swing = np.maximum(0.25 * sn_future, 5.0)
            pred_residuals = np.clip(pred_residuals, -max_swing, max_swing)
        else:
            pred_residuals = np.zeros(len(future_timestamps))

        hybrid_pred = np.clip(sn_future + pred_residuals, 0.0, None)
        low_bound = np.clip(hybrid_pred + empirical_q10, 0.0, None)
        up_bound = np.maximum(hybrid_pred + empirical_q90, hybrid_pred)
        low_bound = np.minimum(low_bound, hybrid_pred)

        future_df["predicted_power_kw"] = np.round(hybrid_pred, 2)
        future_df["lower_bound"] = np.round(low_bound, 2)
        future_df["upper_bound"] = np.round(up_bound, 2)
        future_df["baseline_power_kw"] = np.round(sn_future, 2)

        # 4. Apply Planning Levers to Scenario Curve
        scen_p = hybrid_pred.copy()
        scen_l = low_bound.copy()
        scen_u = up_bound.copy()

        # Occupancy reduction
        if occupancy_reduction_pct > 0:
            occ_fac = 1.0 - (occupancy_reduction_pct / 100.0) * 0.60
            scen_p *= occ_fac
            scen_l *= occ_fac
            scen_u *= occ_fac

        # Solar PV generation
        if solar_pv_kwp > 0:
            hr_series = future_df["timestamp"].dt.hour
            is_sun = hr_series.between(6, 18).astype(float)
            solar_red = is_sun * (solar_pv_kwp * 0.80)
            scen_p = np.clip(scen_p - solar_red, 0.0, None)
            scen_l = np.clip(scen_l - solar_red, 0.0, None)
            scen_u = np.clip(scen_u - solar_red, 0.0, None)

        # LED upgrade (40% saving on ~22% lighting baseload)
        if led_upgrade:
            led_fac = 1.0 - (0.22 * 0.40)
            scen_p *= led_fac
            scen_l *= led_fac
            scen_u *= led_fac

        # AC cutback policy after 18:00
        if ac_policy_reduction_pct > 0:
            hr_series = future_df["timestamp"].dt.hour
            after_6 = (hr_series >= 18).astype(float)
            ac_fac = 1.0 - (ac_policy_reduction_pct / 100.0) * after_6
            scen_p *= ac_fac
            scen_l *= ac_fac
            scen_u *= ac_fac

        scen_l = np.clip(np.minimum(scen_l, scen_p), 0.0, None)
        scen_u = np.maximum(scen_u, scen_p)

        future_df["scenario_power_kw"] = np.round(scen_p, 2)
        future_df["scenario_lower"] = np.round(scen_l, 2)
        future_df["scenario_upper"] = np.round(scen_u, 2)

        if m_id is not None:
            future_df["asset_id"] = m_id

        res_cols = [
            "timestamp",
            "predicted_power_kw",
            "lower_bound",
            "upper_bound",
            "baseline_power_kw",
            "scenario_power_kw",
            "scenario_lower",
            "scenario_upper",
        ]
        if m_id is not None:
            res_cols.insert(1, "asset_id")

        series_list.append(future_df[res_cols])

    if not series_list:
        return pd.DataFrame(columns=empty_cols)

    combined_df = pd.concat(series_list, ignore_index=True)

    if return_campus_total and "asset_id" in combined_df.columns:
        campus_df = combined_df.groupby("timestamp")[
            [
                "predicted_power_kw",
                "lower_bound",
                "upper_bound",
                "baseline_power_kw",
                "scenario_power_kw",
                "scenario_lower",
                "scenario_upper",
            ]
        ].sum().reset_index()
        return campus_df

    return combined_df


