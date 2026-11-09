"""Thermal Comfort Analytics Module (ASHRAE 55 / ISO 7730 PMV & PPD).

Computes Predicted Mean Vote (PMV), Predicted Percentage Dissatisfied (PPD),
and Thom's Discomfort Index for campus zones and buildings.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from campus_energy import assumptions


# Default geographic coordinates for campus buildings
DEFAULT_CAMPUS_COORDS: Dict[str, list[float]] = {
    "BLD-A": [84.8185, 22.2480],
    "BLD-B": [84.8195, 22.2485],
    "BLD-ADM": [84.8192, 22.2472],
    "BLD-LIB": [84.8198, 22.2475],
    "BLD-HST": [84.8175, 22.2460],
    "BLD-UTL": [84.8210, 22.2465],
}

# Sub-zone micro-offsets around building center
DEFAULT_ZONE_OFFSETS: Dict[str, list[float]] = {
    "BLD-A-CR101": [-0.00045, -0.00025],
    "BLD-A-CR102": [-0.00030, 0.00030],
    "BLD-A-CL01": [0.00035, -0.00020],
    "BLD-A-CL02": [0.00040, 0.00028],
    "BLD-B-CR201": [-0.00035, 0.00020],
    "BLD-B-LAB1": [0.00035, -0.00025],
    "BLD-ADM-OF101": [-0.00030, -0.00020],
    "BLD-ADM-MR201": [0.00030, 0.00020],
    "BLD-LIB-RH": [-0.00035, 0.00015],
    "BLD-LIB-CA": [0.00035, -0.00020],
    "BLD-HST-CA": [-0.00030, 0.00020],
    "BLD-HST-UA": [0.00030, -0.00020],
    "BLD-UTL-P1": [-0.00025, 0.00015],
    "BLD-UTL-P2": [0.00025, -0.00015],
}


def calculate_pmv_ppd(
    temp_c: float,
    humidity_pct: float,
    air_velocity_mps: float = 0.15,
    metabolic_rate: float = 1.1,
    clothing_clo: float = 0.6,
) -> Dict[str, Any]:
    """Calculate PMV and PPD using ASHRAE 55 / ISO 7730 Fanger approximation.

    Args:
        temp_c: Dry bulb temperature in Celsius.
        humidity_pct: Relative humidity (0 - 100%).
        air_velocity_mps: Indoor air speed in m/s (default 0.15 m/s).
        metabolic_rate: Metabolic rate in met (default 1.1 for sedentary activity).
        clothing_clo: Clothing insulation in clo (default 0.6 for light summer attire).

    Returns:
        Dict with pmv, ppd, category, color, advisory.
    """
    optimal_setpoint = float(assumptions.get_value("comfort.optimal_setpoint_c"))
    default_air_vel = float(assumptions.get_value("comfort.air_velocity_mps"))
    default_met = float(assumptions.get_value("comfort.metabolic_rate_met"))
    default_clo = float(assumptions.get_value("comfort.clothing_insulation_clo"))

    # Water vapor pressure P_a in hPa (mbar) via Antoine / Tetens approximation
    # Pa = (humidity_pct / 100) * 10 * exp(16.6536 - 4030.183 / (temp_c + 235))
    pa = (humidity_pct / 100.0) * 10.0 * math.exp(16.6536 - 4030.183 / (temp_c + 235.0))

    # Sensitivity factor TS = 0.303 * exp(-0.036 * M) + 0.028
    # For nominal sedentary met:
    ts_ref = 0.303 * math.exp(-0.036 * default_met) + 0.028
    ts = 0.303 * math.exp(-0.036 * metabolic_rate) + 0.028

    # Thermal load L normalized so that at default met:
    # PMV = (temp_c - optimal_setpoint) * 0.28 + (Pa - 16.0) * 0.015
    load = ((temp_c - optimal_setpoint) * 0.28 + (pa - 16.0) * 0.015) / ts_ref

    # Minor velocity and clothing adjustments
    cooling_effect = max(0.0, (air_velocity_mps - default_air_vel) * 1.2)
    clothing_scale = clothing_clo / default_clo if default_clo > 0 else 1.0

    raw_pmv = (ts * load - cooling_effect) * clothing_scale
    pmv = float(np.clip(round(raw_pmv, 2), -3.0, 3.0))

    # PPD (%) = 100 - 95 * exp(-0.03353 * PMV^4 - 0.2179 * PMV^2)
    ppd = round(100.0 - 95.0 * math.exp(-0.03353 * (pmv ** 4) - 0.2179 * (pmv ** 2)), 1)
    ppd = float(np.clip(ppd, 5.0, 100.0))

    # ASHRAE 55 categories
    opt_int = int(round(optimal_setpoint))
    if -0.5 <= pmv <= 0.5:
        category = "Comfortable (Neutral)"
        color = "#10b981"
        advisory = "Setpoint optimal (ASHRAE 55 compliant). Maintain current operation."
    elif 0.5 < pmv <= 1.5:
        category = "Slightly Warm"
        color = "#f59e0b"
        advisory = f"Slightly warm: increase airflow or reduce setpoint to {opt_int}°C."
    elif pmv > 1.5:
        category = "Warm / Heat Discomfort"
        color = "#ef4444"
        advisory = "High thermal discomfort: activate cooling and increase ventilation."
    elif -1.5 <= pmv < -0.5:
        category = "Slightly Cool"
        color = "#38bdf8"
        advisory = f"Increase setpoint to {opt_int}°C to save energy and improve comfort."
    else:
        category = "Cool / Overcooled"
        color = "#6366f1"
        advisory = "Overcooled zone: raise AC setpoint immediately to avoid energy waste."

    return {
        "pmv": pmv,
        "ppd": ppd,
        "category": category,
        "color": color,
        "advisory": advisory,
    }


def compute_pmv(
    temp_c: float,
    humidity_pct: float = 50.0,
    air_velocity_mps: float = 0.15,
    metabolic_rate: float = 1.1,
    clothing_clo: float = 0.6,
) -> float:
    """Compute Predicted Mean Vote (PMV) using ASHRAE 55 / ISO 7730 Fanger model.

    Args:
        temp_c: Dry bulb temperature in Celsius.
        humidity_pct: Relative humidity (0 - 100%, default 50.0%).
        air_velocity_mps: Indoor air speed in m/s (default 0.15 m/s).
        metabolic_rate: Metabolic rate in met (default 1.1 for sedentary).
        clothing_clo: Clothing insulation in clo (default 0.6 for light indoor attire).

    Returns:
        Predicted Mean Vote float clipped to [-3.0, +3.0].
    """
    res = calculate_pmv_ppd(
        temp_c=temp_c,
        humidity_pct=humidity_pct,
        air_velocity_mps=air_velocity_mps,
        metabolic_rate=metabolic_rate,
        clothing_clo=clothing_clo,
    )
    return float(res["pmv"])


def calculate_discomfort_index(temp_c: float, humidity_pct: float) -> Dict[str, Any]:
    """Calculate Thom's Discomfort Index (DI).

    DI = T - 0.55 * (1 - 0.01 * RH) * (T - 14.5)
    """
    di = temp_c - 0.55 * (1.0 - 0.01 * humidity_pct) * (temp_c - 14.5)
    di = round(di, 1)

    if di < 21.0:
        cat = "No Discomfort"
        desc = "100% of occupants feel thermally comfortable"
    elif di < 24.0:
        cat = "Under 50% Discomfort"
        desc = "Less than 50% of occupants feel slight discomfort"
    elif di < 27.0:
        cat = "Over 50% Discomfort"
        desc = "More than 50% of occupants feel discomfort"
    elif di < 29.0:
        cat = "Severe Discomfort"
        desc = "Most occupants feel uncomfortable; cooling recommended"
    else:
        cat = "Heat Stress Risk"
        desc = "State of medical discomfort and heat stress; emergency cooling required"

    return {"di": di, "category": cat, "description": desc}


def get_comfort_snapshot(
    context_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    timestamp: Optional[Any] = None,
    tel_df: Optional[pd.DataFrame] = None,
    campus_coords: Optional[Dict[str, list[float]]] = None,
    zone_offsets: Optional[Dict[str, list[float]]] = None,
) -> pd.DataFrame:
    """Produce a zone-level thermal comfort snapshot at a specific or latest timestamp.

    Args:
        context_df: Context DataFrame containing timestamp, building_id, zone_id,
                    outdoor_temperature_c, humidity_pct.
        assets_df: Assets DataFrame with zone metadata (asset_id, asset_name, zone_id, building_id).
        timestamp: Optional specific timestamp. If None, uses latest in context_df.
        tel_df: Optional telemetry DataFrame to extract measured indoor temperatures from HVAC sensors.
        campus_coords: Optional building coordinate dictionary.
        zone_offsets: Optional zone micro-offset dictionary.

    Returns:
        DataFrame containing building_id, zone_id, zone_name, temperature_c,
        humidity_pct, pmv, ppd, category, color, advisory, lon, lat.
    """
    if context_df.empty:
        return pd.DataFrame(
            columns=[
                "building_id",
                "zone_id",
                "zone_name",
                "temperature_c",
                "humidity_pct",
                "pmv",
                "ppd",
                "category",
                "color",
                "advisory",
                "lon",
                "lat",
            ]
        )

    coords = campus_coords or DEFAULT_CAMPUS_COORDS
    offsets = zone_offsets or DEFAULT_ZONE_OFFSETS

    # Filter by timestamp
    ctx = context_df.copy()
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"])
    if timestamp is not None:
        target_ts = pd.to_datetime(timestamp)
        sub_ctx = ctx[ctx["timestamp"] == target_ts]
        if sub_ctx.empty:
            # Fall back to nearest or latest timestamp
            sub_ctx = ctx[ctx["timestamp"] == ctx["timestamp"].max()]
    else:
        sub_ctx = ctx[ctx["timestamp"] == ctx["timestamp"].max()]

    # Extract zone asset metadata
    zone_assets = assets_df[assets_df["asset_type"] == "zone"].copy()
    if zone_assets.empty:
        # Fall back to unique building/zone pairs in assets
        zone_assets = assets_df.dropna(subset=["zone_id"]).drop_duplicates(subset=["zone_id"]).copy()
        if "asset_name" not in zone_assets.columns:
            zone_assets["asset_name"] = zone_assets["zone_id"]

    # Zone temperature lookup from telemetry if available
    tel_zone_temps: Dict[str, float] = {}
    if tel_df is not None and not tel_df.empty and "temperature_c" in tel_df.columns:
        tel = tel_df.copy()
        tel["timestamp"] = pd.to_datetime(tel["timestamp"])
        tel_ts = target_ts if timestamp is not None else sub_ctx["timestamp"].iloc[0]
        sub_tel = tel[tel["timestamp"] == tel_ts]
        if not sub_tel.empty:
            merged_tel = sub_tel.merge(
                assets_df[["asset_id", "zone_id"]].dropna(subset=["zone_id"]),
                on="asset_id",
                how="inner",
            )
            for _, r in merged_tel.iterrows():
                if pd.notna(r.get("temperature_c")) and r["temperature_c"] > 0:
                    tel_zone_temps[r["zone_id"]] = float(r["temperature_c"])

    # Determine environmental columns in context
    temp_col = "temperature_c" if "temperature_c" in sub_ctx.columns else "outdoor_temperature_c"
    hum_col = "humidity_pct" if "humidity_pct" in sub_ctx.columns else (
        "humidity" if "humidity" in sub_ctx.columns else None
    )

    rows = []
    # Iterate through unique zones found in context or zone_assets
    seen_zones = set()

    for _, r in sub_ctx.iterrows():
        bld_id = str(r["building_id"])
        z_id = str(r["zone_id"]) if pd.notna(r.get("zone_id")) else f"{bld_id}-MAIN"
        if z_id in seen_zones:
            continue
        seen_zones.add(z_id)

        # Temperature: preference order = tel_zone_temps -> sub_ctx[temp_col] -> fallback setpoint
        if z_id in tel_zone_temps:
            t_c = tel_zone_temps[z_id]
        elif temp_col in r and pd.notna(r[temp_col]):
            t_c = float(r[temp_col])
        else:
            t_c = float(assumptions.get_value("comfort.optimal_setpoint_c"))

        # Humidity
        if hum_col and hum_col in r and pd.notna(r[hum_col]):
            h_pct = float(r[hum_col])
        else:
            h_pct = 50.0

        # Zone name lookup
        match = zone_assets[zone_assets["zone_id"] == z_id]
        if not match.empty:
            z_name = str(match.iloc[0]["asset_name"])
        else:
            z_name = z_id.replace("BLD-", "Building ").replace("-", " ")

        # Coordinates
        base_coord = coords.get(bld_id, [84.8192, 22.2472])
        z_offset = offsets.get(z_id, [0.0, 0.0])
        lon = base_coord[0] + z_offset[0]
        lat = base_coord[1] + z_offset[1]

        # Calculate PMV/PPD
        comfort = calculate_pmv_ppd(t_c, h_pct)

        rows.append({
            "building_id": bld_id,
            "zone_id": z_id,
            "zone_name": z_name,
            "temperature_c": round(t_c, 2),
            "humidity_pct": round(h_pct, 1),
            "pmv": comfort["pmv"],
            "ppd": comfort["ppd"],
            "category": comfort["category"],
            "color": comfort["color"],
            "advisory": comfort["advisory"],
            "lon": lon,
            "lat": lat,
        })

    return pd.DataFrame(rows)


def get_campus_comfort_summary(comfort_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate aggregate comfort metrics across campus zones.

    Args:
        comfort_df: Snapshot DataFrame produced by get_comfort_snapshot.

    Returns:
        Dict with avg_temp, avg_humidity, avg_pmv, avg_ppd, pct_compliant.
    """
    if comfort_df.empty:
        return {
            "avg_temp": 0.0,
            "avg_humidity": 0.0,
            "avg_pmv": 0.0,
            "avg_ppd": 0.0,
            "pct_compliant": 0.0,
        }

    avg_temp = round(float(comfort_df["temperature_c"].mean()), 1)
    avg_hum = round(float(comfort_df["humidity_pct"].mean()), 1)
    avg_pmv = round(float(comfort_df["pmv"].mean()), 2)
    avg_ppd = round(float(comfort_df["ppd"].mean()), 1)

    # % Zones compliant with ASHRAE 55 (-0.5 <= PMV <= 0.5)
    compliant_mask = comfort_df["pmv"].abs() <= 0.5
    pct_compliant = round(float(compliant_mask.mean() * 100.0), 1)

    return {
        "avg_temp": avg_temp,
        "avg_humidity": avg_hum,
        "avg_pmv": avg_pmv,
        "avg_ppd": avg_ppd,
        "pct_compliant": pct_compliant,
    }
