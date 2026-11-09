"""Dedicated ESG & Carbon Impact calculator — with BEE EPI scoring."""
import pandas as pd
import numpy as np

# India grid emission factor (kg CO2 per kWh) — CEA 2023 estimate
GRID_EMISSION_FACTOR_KG_PER_KWH = 0.82
# Electricity cost in INR per kWh (approximate commercial tariff in Odisha)
COST_PER_KWH_INR = 8.5
# Solar emission factor (essentially zero for generation, lifecycle ~40g/kWh)
SOLAR_EMISSION_FACTOR_KG_PER_KWH = 0.04

# ─────────────────────────────────────────────────────────────────────────────
# BEE Energy Performance Index (EPI) benchmarks — kWh / m² / year
# Source: Bureau of Energy Efficiency, Ministry of Power, Government of India
# ECBC 2017 + BEE star-rating norms
# ─────────────────────────────────────────────────────────────────────────────
BEE_EPI_BENCHMARKS = {
    "university": {
        "label": "Educational Building",
        "stars": {
            5: (0,    60),    # 5-star: < 60 kWh/m²/yr (excellent)
            4: (60,   90),    # 4-star: 60–90
            3: (90,   120),   # 3-star: 90–120 (ECBC baseline)
            2: (120,  160),   # 2-star: 120–160
            1: (160,  9999),  # 1-star: > 160 (poor)
        },
        "ecbc_baseline": 100,  # kWh/m²/yr
    },
    "hospital": {
        "label": "Healthcare Facility",
        "stars": {
            5: (0,    150),
            4: (150,  250),
            3: (250,  350),   # BEE baseline for hospitals
            2: (350,  500),
            1: (500,  9999),
        },
        "ecbc_baseline": 300,
    },
    "govt_office": {
        "label": "Government Office",
        "stars": {
            5: (0,    80),
            4: (80,   120),
            3: (120,  160),   # ECBC baseline for offices
            2: (160,  220),
            1: (220,  9999),
        },
        "ecbc_baseline": 140,
    },
    "industrial": {
        "label": "Industrial Facility",
        "stars": {
            5: (0,    200),
            4: (200,  350),
            3: (350,  500),
            2: (500,  700),
            1: (700,  9999),
        },
        "ecbc_baseline": 450,
    },
}

# NAPCC (National Action Plan on Climate Change) — Perform Achieve Trade (PAT)
# Recommended 20% reduction from baseline for large energy consumers
NAPCC_REDUCTION_TARGET_PCT = 20.0


def _get_bee_star(epi_annual: float, facility_type: str) -> dict:
    """Return BEE star rating and details for a given annual EPI."""
    norms = BEE_EPI_BENCHMARKS.get(facility_type, BEE_EPI_BENCHMARKS["university"])
    stars = 1
    for star, (lo, hi) in norms["stars"].items():
        if lo <= epi_annual < hi:
            stars = star
            break
    ecbc_baseline = norms["ecbc_baseline"]
    vs_ecbc_pct = round(((epi_annual - ecbc_baseline) / ecbc_baseline) * 100, 1)
    napcc_target = round(ecbc_baseline * (1 - NAPCC_REDUCTION_TARGET_PCT / 100), 1)
    return {
        "stars": stars,
        "epi_annual_kwh_m2": round(epi_annual, 1),
        "ecbc_baseline_kwh_m2": ecbc_baseline,
        "vs_ecbc_pct": vs_ecbc_pct,            # negative = better than baseline
        "napcc_target_kwh_m2": napcc_target,
        "facility_label": norms["label"],
        "compliant": epi_annual <= ecbc_baseline,
    }


def calculate_esg_metrics(
    telemetry_df: pd.DataFrame,
    events_df: pd.DataFrame,
    floor_area_m2: float = 10000.0,
    facility_type: str = "university",
) -> dict:
    """Compute campus-wide ESG metrics from telemetry and injected events.

    Args:
        telemetry_df: Full telemetry dataframe.
        events_df: Injected/detected events dataframe.
        floor_area_m2: Total conditioned floor area of the facility.
        facility_type: One of 'university', 'hospital', 'govt_office', 'industrial'.

    Returns a dict with top-level KPIs, BEE rating, and daily/building breakdowns.
    """
    tel = telemetry_df.copy()
    tel["timestamp"] = pd.to_datetime(tel["timestamp"], utc=True)
    tel["date"] = tel["timestamp"].dt.date

    # --- Total energy from main building meters ---
    meters = tel[tel["reading_type"] == "meter"].copy()
    total_energy_kwh = meters["interval_energy_kwh"].sum()
    total_cost_inr = total_energy_kwh * COST_PER_KWH_INR
    total_carbon_kg = total_energy_kwh * GRID_EMISSION_FACTOR_KG_PER_KWH

    # --- Avoidable waste: estimate 80% of anomalous consumption as avoidable ---
    evt = events_df.copy()
    evt["start_time"] = pd.to_datetime(evt["start_time"], utc=True)
    evt["end_time"] = pd.to_datetime(evt["end_time"], utc=True)

    wasted_kwh = 0.0
    for _, row in evt.iterrows():
        mask = (
            (tel["asset_id"] == row["asset_id"])
            & (tel["timestamp"] >= row["start_time"])
            & (tel["timestamp"] <= row["end_time"])
        )
        wasted_kwh += tel.loc[mask, "interval_energy_kwh"].sum() * 0.80

    avoidable_cost_inr = wasted_kwh * COST_PER_KWH_INR
    avoidable_carbon_kg = wasted_kwh * GRID_EMISSION_FACTOR_KG_PER_KWH

    # --- Daily campus energy trend ---
    daily = (
        meters.groupby("date")["interval_energy_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"interval_energy_kwh": "daily_kwh"})
    )
    daily["date"] = daily["date"].astype(str)
    daily["daily_cost_inr"] = daily["daily_kwh"] * COST_PER_KWH_INR
    daily["daily_carbon_kg"] = daily["daily_kwh"] * GRID_EMISSION_FACTOR_KG_PER_KWH

    # --- Per-building breakdown ---
    building_id_col = "asset_id"
    building_energy = (
        meters.groupby(building_id_col)["interval_energy_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"interval_energy_kwh": "total_kwh"})
    )
    building_energy["cost_inr"] = building_energy["total_kwh"] * COST_PER_KWH_INR
    building_energy["carbon_kg"] = building_energy["total_kwh"] * GRID_EMISSION_FACTOR_KG_PER_KWH
    building_energy["building_name"] = building_energy[building_id_col].str.replace("MTR-", "")

    # --- Annualised projections (28-day window → extrapolate to 365) ---
    simulation_days = max(len(daily), 1)
    annual_kwh = (total_energy_kwh / simulation_days) * 365
    annual_cost_inr = annual_kwh * COST_PER_KWH_INR
    annual_carbon_tons = (annual_kwh * GRID_EMISSION_FACTOR_KG_PER_KWH) / 1000

    # --- BEE EPI Score ---
    epi_annual = annual_kwh / max(floor_area_m2, 1.0)
    bee = _get_bee_star(epi_annual, facility_type)

    # Trees equivalent (1 tree absorbs ~21 kg CO₂/year)
    trees_equivalent = int(annual_carbon_tons * 1000 / 21)
    # NAPCC savings potential
    napcc_savings_kwh = max(0, annual_kwh - (bee["napcc_target_kwh_m2"] * floor_area_m2))
    napcc_savings_cost = napcc_savings_kwh * COST_PER_KWH_INR
    napcc_savings_carbon_tons = (napcc_savings_kwh * GRID_EMISSION_FACTOR_KG_PER_KWH) / 1000

    return {
        "kpis": {
            "total_energy_kwh": round(total_energy_kwh, 1),
            "total_cost_inr": round(total_cost_inr, 2),
            "total_carbon_kg": round(total_carbon_kg, 1),
            "total_carbon_tons": round(total_carbon_kg / 1000, 2),
            "wasted_kwh": round(wasted_kwh, 1),
            "avoidable_cost_inr": round(avoidable_cost_inr, 2),
            "avoidable_carbon_kg": round(avoidable_carbon_kg, 1),
            "waste_pct": round((wasted_kwh / total_energy_kwh * 100) if total_energy_kwh > 0 else 0, 1),
            "annual_kwh_projection": round(annual_kwh, 0),
            "annual_cost_inr_projection": round(annual_cost_inr, 0),
            "annual_carbon_tons_projection": round(annual_carbon_tons, 2),
            "trees_equivalent": trees_equivalent,
            "napcc_savings_kwh": round(napcc_savings_kwh, 0),
            "napcc_savings_cost_inr": round(napcc_savings_cost, 0),
            "napcc_savings_carbon_tons": round(napcc_savings_carbon_tons, 2),
        },
        "bee": bee,
        "daily_trend": daily,
        "building_breakdown": building_energy,
    }
