"""Water Resource Simulation & Analytics Module.

Simulates building-level water consumption, flow rates (LPM),
Minimum Night Flow (MNF) baselines, and leak anomalies (overnight pipe leaks, bursts).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WATER_TARIFF_INR_PER_KL = 25.0  # ₹25 per kilo-liter (1,000 Liters)


def _classify_building(building_id: str, building_name: str) -> str:
    """Classify building into profile type based on ID or name keywords."""
    combined = f"{building_id} {building_name}".lower()
    if any(k in combined for k in ["hst", "hostel", "ward", "ipd", "res", "residential", "inpatient"]):
        return "residential"
    if any(k in combined for k in ["utl", "utility", "prod", "production", "plant", "shopfloor", "industrial", "pump"]):
        return "industrial"
    return "academic_office"


def _get_base_profile(bld_type: str, hour: int, is_weekend: bool) -> float:
    """Return typical baseline flow in Liters per hour."""
    if bld_type == "residential":
        # 24/7 occupancy with morning and evening peaks
        if hour in (1, 2, 3, 4):
            return 140.0  # ~2.3 LPM night baseline
        if hour in (0, 5):
            return 320.0
        if 6 <= hour <= 8:
            return 3200.0  # morning peak ~53 LPM
        if 9 <= hour <= 11:
            return 1600.0
        if 12 <= hour <= 14:
            return 1900.0  # lunch peak
        if 15 <= hour <= 18:
            return 1500.0
        if 19 <= hour <= 21:
            return 2900.0  # evening peak ~48 LPM
        return 650.0  # late evening

    if bld_type == "industrial":
        # Multi-shift industrial or plant utilities
        if hour in (1, 2, 3, 4):
            return 350.0  # night baseline ~5.8 LPM
        if 6 <= hour <= 14:
            base = 2400.0  # Shift 1 (~40 LPM)
        elif 15 <= hour <= 22:
            base = 1900.0  # Shift 2 (~32 LPM)
        else:
            base = 600.0  # Transition / night shift
        return base * (0.4 if is_weekend else 1.0)

    # academic_office
    if hour in (1, 2, 3, 4):
        return 30.0  # minimal night flow ~0.5 LPM
    if hour in (0, 5, 6, 22, 23):
        return 60.0
    if hour == 7:
        return 250.0
    if hour == 8:
        return 800.0
    if 9 <= hour <= 11:
        base = 1600.0
    elif 12 <= hour <= 14:
        base = 2200.0  # Lunch / cafeteria peak
    elif 15 <= hour <= 17:
        base = 1500.0
    elif hour == 18:
        base = 700.0
    elif 19 <= hour <= 21:
        base = 200.0
    else:
        base = 100.0

    if is_weekend:
        base *= 0.15  # minimal activity on weekends
    return base


def generate_water_data(
    assets_df: pd.DataFrame,
    days: int = 28,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic hourly water telemetry for campus buildings.

    Args:
        assets_df: Assets dataframe with building rows.
        days: Duration in days (default 28).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns:
        [timestamp, building_id, water_meter_id, flow_liters, flow_rate_lpm, is_anomaly, anomaly_type]
    """
    rng = np.random.default_rng(seed)

    # Extract building IDs
    buildings: List[str] = []
    if "asset_type" in assets_df.columns:
        buildings = assets_df[assets_df["asset_type"] == "building"]["asset_id"].dropna().tolist()
    if not buildings and "building_id" in assets_df.columns:
        buildings = [str(b) for b in assets_df["building_id"].dropna().unique().tolist() if str(b)]
    if not buildings:
        buildings = ["BLD-MAIN"]

    # Name mapping for keyword classification
    name_map: Dict[str, str] = {}
    if "asset_name" in assets_df.columns and "asset_id" in assets_df.columns:
        for _, row in assets_df[assets_df["asset_id"].isin(buildings)].iterrows():
            name_map[str(row["asset_id"])] = str(row["asset_name"])

    # Build building classifications
    bld_types = {bld: _classify_building(bld, name_map.get(bld, "")) for bld in buildings}

    start_date = pd.Timestamp("2025-09-01 00:00:00")
    total_hours = days * 24
    timestamps = [start_date + timedelta(hours=i) for i in range(total_hours)]

    rows: List[Dict[str, Any]] = []

    for bld in buildings:
        bld_type = bld_types[bld]
        meter_id = f"WM-{bld}"

        for i, ts in enumerate(timestamps):
            hour = ts.hour
            is_weekend = ts.weekday() >= 5
            day_idx = i // 24

            base = _get_base_profile(bld_type, hour, is_weekend)

            # Apply random variation (jitter ~ +/- 8%)
            noise_factor = rng.uniform(0.92, 1.08)
            additive_noise = rng.normal(0, 10.0)
            flow = max(0.0, base * noise_factor + additive_noise)

            is_anomaly = False
            anomaly_type = "normal"

            # Inject Leak 1: Overnight pipe leak in 1st building from Day 10 to Day 14 (1 AM - 5 AM)
            # +35 L/min = +2100 L/hr
            if bld == buildings[0] and (10 <= day_idx <= 14) and (1 <= hour <= 4):
                flow += 2100.0 + rng.uniform(-40.0, 40.0)
                is_anomaly = True
                anomaly_type = "pipe_leak_overnight"

            # Inject Leak 2: Sudden burst / valve leak in 2nd building on Day 20 for 6 hours
            # +5000 L/hr (+83.3 L/min) between 10:00 and 16:00
            burst_building = buildings[1 % len(buildings)]
            if bld == burst_building and day_idx == 20 and (10 <= hour <= 15):
                flow += 5000.0 + rng.uniform(-80.0, 80.0)
                is_anomaly = True
                anomaly_type = "burst_valve_leak"

            flow_liters = round(flow, 1)
            flow_rate_lpm = round(flow_liters / 60.0, 2)

            rows.append({
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "building_id": bld,
                "water_meter_id": meter_id,
                "flow_liters": flow_liters,
                "flow_rate_lpm": flow_rate_lpm,
                "is_anomaly": is_anomaly,
                "anomaly_type": anomaly_type,
            })

    df = pd.DataFrame(rows)
    return df


def analyze_water_data(water_df: pd.DataFrame) -> Dict[str, Any]:
    """Perform water consumption analytics, MNF baseline, and leak diagnosis.

    Args:
        water_df: Water telemetry DataFrame with columns:
            [timestamp, building_id, water_meter_id, flow_liters, flow_rate_lpm, is_anomaly, anomaly_type]

    Returns:
        Dict with keys:
            - kpis: Dictionary of summary KPIs
            - daily_trend: DataFrame with daily consumption, cost, and 3-day MA
            - building_breakdown: DataFrame of per-building consumption & share
            - diurnal_curve: DataFrame of 24h average hourly curve with MNF zone
            - leak_events: DataFrame / list of detected leak events & recommendations
    """
    if water_df.empty:
        return {
            "kpis": {
                "total_water_kl": 0.0,
                "water_cost_inr": 0.0,
                "active_leaks": 0,
                "avoidable_loss_kl": 0.0,
                "avoidable_loss_inr": 0.0,
                "night_leak_lpm": 0.0,
                "daily_avg_kl": 0.0,
                "cost_per_kl": WATER_TARIFF_INR_PER_KL,
            },
            "daily_trend": pd.DataFrame(),
            "building_breakdown": pd.DataFrame(),
            "diurnal_curve": pd.DataFrame(),
            "leak_events": pd.DataFrame(),
        }

    df = water_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    # 1. Total Water and Baseline Calculation
    total_flow_liters = float(df["flow_liters"].sum())
    total_water_kl = round(total_flow_liters / 1000.0, 1)
    water_cost_inr = round(total_water_kl * WATER_TARIFF_INR_PER_KL, 0)

    # Calculate hourly baseline per building (using non-anomaly rows where possible)
    normal_df = df[~df["is_anomaly"]] if ("is_anomaly" in df.columns and df["is_anomaly"].any()) else df
    baseline_lookup = normal_df.groupby(["building_id", "hour"])["flow_liters"].median().to_dict()

    # Avoidable water loss (excess over baseline during anomalies)
    avoidable_loss_liters = 0.0
    night_leak_lpm_vals: List[float] = []

    if "is_anomaly" in df.columns and df["is_anomaly"].any():
        for _, row in df[df["is_anomaly"]].iterrows():
            bld = row["building_id"]
            hr = row["hour"]
            base = baseline_lookup.get((bld, hr), 0.0)
            excess = max(0.0, row["flow_liters"] - base)
            avoidable_loss_liters += excess
            if hr in (1, 2, 3, 4):
                night_leak_lpm_vals.append(excess / 60.0)

    avoidable_loss_kl = round(avoidable_loss_liters / 1000.0, 1)
    avoidable_loss_inr = round(avoidable_loss_kl * WATER_TARIFF_INR_PER_KL, 0)
    night_leak_lpm = round(float(np.mean(night_leak_lpm_vals)), 1) if night_leak_lpm_vals else 0.0

    # 2. Daily Trend
    daily = (
        df.groupby("date")
        .agg(
            flow_liters=("flow_liters", "sum"),
            anomalies=("is_anomaly", "sum") if "is_anomaly" in df.columns else ("flow_liters", lambda _: 0),
        )
        .reset_index()
    )
    daily["flow_kl"] = (daily["flow_liters"] / 1000.0).round(2)
    daily["cost_inr"] = (daily["flow_kl"] * WATER_TARIFF_INR_PER_KL).round(0)
    daily["rolling_avg_3d"] = daily["flow_kl"].rolling(window=3, min_periods=1).mean().round(2)

    # 3. Building Breakdown
    building_breakdown = (
        df.groupby("building_id")
        .agg(
            flow_liters=("flow_liters", "sum"),
            anomalies=("is_anomaly", "sum") if "is_anomaly" in df.columns else ("flow_liters", lambda _: 0),
        )
        .reset_index()
    )
    building_breakdown["total_kl"] = (building_breakdown["flow_liters"] / 1000.0).round(2)
    building_breakdown["cost_inr"] = (building_breakdown["total_kl"] * WATER_TARIFF_INR_PER_KL).round(0)
    if total_water_kl > 0:
        building_breakdown["share_pct"] = ((building_breakdown["total_kl"] / total_water_kl) * 100).round(1)
    else:
        building_breakdown["share_pct"] = 0.0

    # 4. Diurnal Flow Curve (24-hour average hourly profile across campus)
    diurnal_curve = (
        df.groupby("hour")
        .agg(
            avg_flow_liters=("flow_liters", "mean"),
            avg_flow_rate_lpm=("flow_rate_lpm", "mean"),
        )
        .reset_index()
    )
    # Also normal baseline without anomalies
    norm_diurnal = (
        normal_df.groupby("hour")
        .agg(
            baseline_flow_liters=("flow_liters", "mean"),
            baseline_flow_lpm=("flow_rate_lpm", "mean"),
        )
        .reset_index()
    )
    diurnal_curve = pd.merge(diurnal_curve, norm_diurnal, on="hour", how="left")
    diurnal_curve["is_mnf_zone"] = diurnal_curve["hour"].isin([1, 2, 3, 4])
    diurnal_curve["avg_flow_liters"] = diurnal_curve["avg_flow_liters"].round(1)
    diurnal_curve["avg_flow_rate_lpm"] = diurnal_curve["avg_flow_rate_lpm"].round(2)
    diurnal_curve["baseline_flow_lpm"] = diurnal_curve["baseline_flow_lpm"].round(2)

    # 5. Detect Leak Events & Build Remediation Queue
    # We check:
    # a) Night flow (1-5 AM) > 3x typical median night flow
    # b) Any burst events
    leak_records: List[Dict[str, Any]] = []

    # Check for night flow leaks per building
    night_mask = df["hour"].isin([1, 2, 3, 4])
    night_df = df[night_mask]

    for bld in df["building_id"].unique():
        bld_night = night_df[night_df["building_id"] == bld]
        bld_norm_night = normal_df[(normal_df["building_id"] == bld) & (normal_df["hour"].isin([1, 2, 3, 4]))]
        median_mnf = float(bld_norm_night["flow_liters"].median()) if not bld_norm_night.empty else 50.0

        # Group by date to find days with night flow > 3x median_mnf
        night_by_date = bld_night.groupby("date").agg(
            night_avg_flow=("flow_liters", "mean"),
            night_total_flow=("flow_liters", "sum"),
            is_anomaly_sum=("is_anomaly", "sum") if "is_anomaly" in df.columns else ("flow_liters", lambda _: 0),
        ).reset_index()

        leak_days = night_by_date[
            (night_by_date["night_avg_flow"] > max(3.0 * median_mnf, 500.0))
            | (night_by_date["is_anomaly_sum"] > 0)
        ]

        if not leak_days.empty:
            start_d = leak_days["date"].min()
            end_d = leak_days["date"].max()
            dur_hrs = len(leak_days) * 4  # 4 hours per night
            total_leak_vol = max(0.0, float(leak_days["night_total_flow"].sum()) - (median_mnf * dur_hrs))
            leak_kl = round(total_leak_vol / 1000.0, 1)
            cost_loss = round(leak_kl * WATER_TARIFF_INR_PER_KL, 0)
            avg_lpm = round((total_leak_vol / (dur_hrs * 60.0)), 1) if dur_hrs > 0 else 35.0

            leak_records.append({
                "leak_id": f"LK-{bld[:6]}-MNF",
                "building": bld,
                "building_id": bld,
                "leak_type": "Overnight Pipe Leak (MNF > 3x Baseline)",
                "start_time": f"{start_d} 01:00",
                "end_time": f"{end_d} 05:00",
                "duration": f"{dur_hrs}h (across {len(leak_days)} nights)",
                "duration_hours": dur_hrs,
                "leak_rate_lpm": avg_lpm,
                "lost_volume": f"{leak_kl:,.1f} kL",
                "lost_volume_kl": leak_kl,
                "cost_loss_inr": cost_loss,
                "severity": "High",
                "status": "Active Leak",
                "recommendation": f"Inspect supply riser, pressure reducing valve (PRV), and ground joints in {bld} basement; potential concealed pipe fissure.",
            })

    # Also detect daytime bursts / valve leaks
    burst_rows = df[df["anomaly_type"] == "burst_valve_leak"] if "anomaly_type" in df.columns else pd.DataFrame()
    if not burst_rows.empty:
        for bld in burst_rows["building_id"].unique():
            bld_burst = burst_rows[burst_rows["building_id"] == bld]
            start_ts = bld_burst["timestamp"].min()
            end_ts = bld_burst["timestamp"].max()
            dur_hrs = len(bld_burst)
            burst_liters = float(bld_burst["flow_liters"].sum())
            base_liters = sum(baseline_lookup.get((bld, r.hour), 0.0) for r in bld_burst["timestamp"])
            excess = max(0.0, burst_liters - base_liters)
            burst_kl = round(excess / 1000.0, 1)
            cost_loss = round(burst_kl * WATER_TARIFF_INR_PER_KL, 0)
            avg_lpm = round((excess / (dur_hrs * 60.0)), 1) if dur_hrs > 0 else 83.3

            leak_records.append({
                "leak_id": f"LK-{bld[:6]}-BRST",
                "building": bld,
                "building_id": bld,
                "leak_type": "Sudden Valve Burst / Overflow",
                "start_time": start_ts.strftime("%Y-%m-%d %H:%M"),
                "end_time": end_ts.strftime("%Y-%m-%d %H:%M"),
                "duration": f"{dur_hrs}h continuous",
                "duration_hours": dur_hrs,
                "leak_rate_lpm": avg_lpm,
                "lost_volume": f"{burst_kl:,.1f} kL",
                "lost_volume_kl": burst_kl,
                "cost_loss_inr": cost_loss,
                "severity": "Critical",
                "status": "Repaired / Isolated",
                "recommendation": f"Emergency solenoid shutoff triggered. Replace damaged seal ring & recalibrate flow pressure sensor for {bld}.",
            })

    leak_events_df = pd.DataFrame(leak_records)

    # Number of active leaks
    active_leaks_count = len([e for e in leak_records if "Active" in e.get("status", "")])
    if active_leaks_count == 0 and leak_records:
        active_leaks_count = len(leak_records)

    kpis = {
        "total_water_kl": total_water_kl,
        "water_cost_inr": water_cost_inr,
        "active_leaks": active_leaks_count,
        "avoidable_loss_kl": avoidable_loss_kl,
        "avoidable_loss_inr": avoidable_loss_inr,
        "night_leak_lpm": night_leak_lpm,
        "daily_avg_kl": round(total_water_kl / max(len(daily), 1), 1),
        "cost_per_kl": WATER_TARIFF_INR_PER_KL,
    }

    return {
        "kpis": kpis,
        "daily_trend": daily,
        "building_breakdown": building_breakdown,
        "diurnal_curve": diurnal_curve,
        "leak_events": leak_events_df,
    }
