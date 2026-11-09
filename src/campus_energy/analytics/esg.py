"""Dedicated ESG & Carbon Impact calculator — with BEE EPI scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd

from campus_energy import assumptions

# Dynamic factors loaded from centralized assumptions registry
GRID_EMISSION_FACTOR_KG_PER_KWH: float = assumptions.get_value("emissions.grid_factor")
COST_PER_KWH_INR: float = assumptions.get_value("tariff.electricity_commercial")
SOLAR_EMISSION_FACTOR_KG_PER_KWH: float = assumptions.get_value("emissions.solar_lifecycle_factor")
NAPCC_REDUCTION_TARGET_PCT: float = assumptions.get_value("policy.napcc_reduction_target_pct")

# ─────────────────────────────────────────────────────────────────────────────
# BEE Energy Performance Index (EPI) benchmarks — kWh / m² / year
# Source: Bureau of Energy Efficiency, Ministry of Power, Government of India
# ECBC 2017 + BEE star-rating norms (loaded from assumptions registry)
# ─────────────────────────────────────────────────────────────────────────────
BEE_EPI_BENCHMARKS = {
    "university": {
        "label": "Educational Building",
        "stars": {
            5: (0, assumptions.get_value("bee.epi.university.five_star_max")),
            4: (assumptions.get_value("bee.epi.university.five_star_max"), assumptions.get_value("bee.epi.university.four_star_max")),
            3: (assumptions.get_value("bee.epi.university.four_star_max"), assumptions.get_value("bee.epi.university.three_star_max")),
            2: (assumptions.get_value("bee.epi.university.three_star_max"), assumptions.get_value("bee.epi.university.two_star_max")),
            1: (assumptions.get_value("bee.epi.university.two_star_max"), 9999),
        },
        "ecbc_baseline": assumptions.get_value("bee.epi.university.ecbc_baseline"),
    },
    "hospital": {
        "label": "Healthcare Facility",
        "stars": {
            5: (0, assumptions.get_value("bee.epi.hospital.five_star_max")),
            4: (assumptions.get_value("bee.epi.hospital.five_star_max"), assumptions.get_value("bee.epi.hospital.four_star_max")),
            3: (assumptions.get_value("bee.epi.hospital.four_star_max"), assumptions.get_value("bee.epi.hospital.three_star_max")),
            2: (assumptions.get_value("bee.epi.hospital.three_star_max"), assumptions.get_value("bee.epi.hospital.two_star_max")),
            1: (assumptions.get_value("bee.epi.hospital.two_star_max"), 9999),
        },
        "ecbc_baseline": assumptions.get_value("bee.epi.hospital.ecbc_baseline"),
    },
    "govt_office": {
        "label": "Government Office",
        "stars": {
            5: (0, assumptions.get_value("bee.epi.govt_office.five_star_max")),
            4: (assumptions.get_value("bee.epi.govt_office.five_star_max"), assumptions.get_value("bee.epi.govt_office.four_star_max")),
            3: (assumptions.get_value("bee.epi.govt_office.four_star_max"), assumptions.get_value("bee.epi.govt_office.three_star_max")),
            2: (assumptions.get_value("bee.epi.govt_office.three_star_max"), assumptions.get_value("bee.epi.govt_office.two_star_max")),
            1: (assumptions.get_value("bee.epi.govt_office.two_star_max"), 9999),
        },
        "ecbc_baseline": assumptions.get_value("bee.epi.govt_office.ecbc_baseline"),
    },
    "industrial": {
        "label": "Industrial Facility",
        "stars": {
            5: (0, assumptions.get_value("bee.epi.industrial.five_star_max")),
            4: (assumptions.get_value("bee.epi.industrial.five_star_max"), assumptions.get_value("bee.epi.industrial.four_star_max")),
            3: (assumptions.get_value("bee.epi.industrial.four_star_max"), assumptions.get_value("bee.epi.industrial.three_star_max")),
            2: (assumptions.get_value("bee.epi.industrial.three_star_max"), assumptions.get_value("bee.epi.industrial.two_star_max")),
            1: (assumptions.get_value("bee.epi.industrial.two_star_max"), 9999),
        },
        "ecbc_baseline": assumptions.get_value("bee.epi.industrial.ecbc_baseline"),
    },
}


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
    cost_per_kwh = assumptions.get_value("tariff.electricity_commercial")
    grid_emission_factor = assumptions.get_value("emissions.grid_factor")
    fault_waste_fraction = assumptions.get_value("waste.fault_avoidable_fraction")
    tree_seq_kg_yr = assumptions.get_value("sustainability.tree_sequestration_kg_co2_yr")

    tel = telemetry_df.copy()
    tel["timestamp"] = pd.to_datetime(tel["timestamp"], utc=True)
    tel["date"] = tel["timestamp"].dt.date

    # --- Total energy from main building meters ---
    meters = tel[tel["reading_type"] == "meter"].copy()
    total_energy_kwh = meters["interval_energy_kwh"].sum()
    total_cost_inr = total_energy_kwh * cost_per_kwh
    total_carbon_kg = total_energy_kwh * grid_emission_factor

    # --- Avoidable waste: estimate fault-waste fraction of anomalous consumption as avoidable ---
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
        wasted_kwh += tel.loc[mask, "interval_energy_kwh"].sum() * fault_waste_fraction

    avoidable_cost_inr = wasted_kwh * cost_per_kwh
    avoidable_carbon_kg = wasted_kwh * grid_emission_factor

    # Standardized Savings Ranges {low, expected, high}
    avoidable_savings_range = {
        "kwh": {
            "low": round(wasted_kwh * 0.80, 1),
            "expected": round(wasted_kwh, 1),
            "high": round(wasted_kwh * 1.25, 1),
        },
        "cost_inr": {
            "low": round(avoidable_cost_inr * 0.80, 2),
            "expected": round(avoidable_cost_inr, 2),
            "high": round(avoidable_cost_inr * 1.25, 2),
        },
        "carbon_kg": {
            "low": round(avoidable_carbon_kg * 0.80, 1),
            "expected": round(avoidable_carbon_kg, 1),
            "high": round(avoidable_carbon_kg * 1.25, 1),
        },
    }

    # --- Daily campus energy trend ---
    daily = (
        meters.groupby("date")["interval_energy_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"interval_energy_kwh": "daily_kwh"})
    )
    daily["date"] = daily["date"].astype(str)
    daily["daily_cost_inr"] = daily["daily_kwh"] * cost_per_kwh
    daily["daily_carbon_kg"] = daily["daily_kwh"] * grid_emission_factor

    # --- Per-building breakdown ---
    building_id_col = "asset_id"
    building_energy = (
        meters.groupby(building_id_col)["interval_energy_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"interval_energy_kwh": "total_kwh"})
    )
    building_energy["cost_inr"] = building_energy["total_kwh"] * cost_per_kwh
    building_energy["carbon_kg"] = building_energy["total_kwh"] * grid_emission_factor
    building_energy["building_name"] = building_energy[building_id_col].str.replace("MTR-", "")

    # --- Annualised projections (28-day window → extrapolate to 365) ---
    simulation_days = max(len(daily), 1)
    annual_kwh = (total_energy_kwh / simulation_days) * 365
    annual_cost_inr = annual_kwh * cost_per_kwh
    annual_carbon_tons = (annual_kwh * grid_emission_factor) / 1000

    # --- BEE EPI Score ---
    epi_annual = annual_kwh / max(floor_area_m2, 1.0)
    bee = _get_bee_star(epi_annual, facility_type)

    # Trees equivalent
    trees_equivalent = int(annual_carbon_tons * 1000 / tree_seq_kg_yr)
    # NAPCC savings potential
    napcc_savings_kwh = max(0, annual_kwh - (bee["napcc_target_kwh_m2"] * floor_area_m2))
    napcc_savings_cost = napcc_savings_kwh * cost_per_kwh
    napcc_savings_carbon_tons = (napcc_savings_kwh * grid_emission_factor) / 1000

    napcc_savings_range = {
        "kwh": {
            "low": round(napcc_savings_kwh * 0.85, 0),
            "expected": round(napcc_savings_kwh, 0),
            "high": round(napcc_savings_kwh * 1.15, 0),
        },
        "cost_inr": {
            "low": round(napcc_savings_cost * 0.85, 0),
            "expected": round(napcc_savings_cost, 0),
            "high": round(napcc_savings_cost * 1.15, 0),
        },
        "carbon_tons": {
            "low": round(napcc_savings_carbon_tons * 0.85, 2),
            "expected": round(napcc_savings_carbon_tons, 2),
            "high": round(napcc_savings_carbon_tons * 1.15, 2),
        },
    }

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
            "avoidable_savings_range": avoidable_savings_range,
            "napcc_savings_range": napcc_savings_range,
            "avoidable_cost_display": f"Expected: ₹{avoidable_cost_inr:,.0f} (Range: ₹{avoidable_cost_inr*0.8:,.0f} – ₹{avoidable_cost_inr*1.25:,.0f})",
            "napcc_savings_cost_display": f"Expected: ₹{napcc_savings_cost:,.0f} (Range: ₹{napcc_savings_cost*0.85:,.0f} – ₹{napcc_savings_cost*1.15:,.0f})",
            "disclaimer": "⚠️ Decision-support estimate, not an official measurement.",
        },
        "bee": bee,
        "daily_trend": daily,
        "building_breakdown": building_energy,
    }


def format_savings_range_inr(range_dict: dict[str, float]) -> str:
    """Format an INR savings range dict {low, expected, high} for UI display."""
    return f"Expected: ₹{range_dict['expected']:,.0f} (Range: ₹{range_dict['low']:,.0f} – ₹{range_dict['high']:,.0f})"


def format_savings_range_kwh(range_dict: dict[str, float]) -> str:
    """Format a kWh savings range dict {low, expected, high} for UI display."""
    return f"{range_dict['expected']:,.1f} kWh [{range_dict['low']:,.1f} – {range_dict['high']:,.1f}]"

