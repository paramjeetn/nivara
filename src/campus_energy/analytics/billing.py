"""Department-level billing and chargeback engine.

Each asset is owned by a department. This module allocates metered energy
to departments using sub-meter telemetry (not naive floor-area splits).
"""
import pandas as pd
import numpy as np

from campus_energy import assumptions

COST_PER_KWH_INR: float = assumptions.get_value("tariff.electricity_commercial")

# Asset → Department mapping based on campus config.
# In a real system this would come from the CMMS / asset registry.
ASSET_DEPARTMENT_MAP = {
    # Academic Block A
    "EQ-A-CR101-LT": "Civil Engineering",
    "EQ-A-CR101-AC": "Civil Engineering",
    "EQ-A-CR102-LT": "Electrical Engineering",
    "EQ-A-CR102-AC": "Electrical Engineering",
    "EQ-A-CL01-LT": "Computer Science",
    "EQ-A-CL01-AC": "Computer Science",
    "EQ-A-CL01-PC": "Computer Science",
    "EQ-A-CL02-LT": "Computer Science",
    "EQ-A-CL02-AC": "Computer Science",
    "EQ-A-CL02-PC": "Computer Science",
    # Academic Block B
    "EQ-B-CR201-LT": "Mechanical Engineering",
    "EQ-B-CR201-AC": "Mechanical Engineering",
    "EQ-B-LAB1-LT": "Mechanical Engineering",
    "EQ-B-LAB1-AC": "Mechanical Engineering",
    "EQ-B-LAB1-EQ": "Mechanical Engineering",
    # Admin
    "EQ-ADM-OF101-LT": "Administration",
    "EQ-ADM-OF101-AC": "Administration",
    "EQ-ADM-OF101-PC": "Administration",
    "EQ-ADM-MR201-LT": "Administration",
    "EQ-ADM-MR201-AC": "Administration",
    # Library
    "EQ-LIB-RH-LT": "Library",
    "EQ-LIB-RH-AC": "Library",
    "EQ-LIB-CA-LT": "Library",
    "EQ-LIB-CA-AC": "Library",
    "EQ-LIB-CA-PC": "Library",
    # Hostel
    "EQ-HST-CA-LT": "Student Affairs (Hostel)",
    "EQ-HST-CA-AC": "Student Affairs (Hostel)",
    "EQ-HST-UA-LT": "Student Affairs (Hostel)",
    # Utility
    "EQ-UTL-P1-PM": "Facilities & Infrastructure",
    "EQ-UTL-P2-PM": "Facilities & Infrastructure",
}

BUILDING_METER_MAP = {
    "MTR-BLD-A": "Academic Block A",
    "MTR-BLD-B": "Academic Block B",
    "MTR-BLD-ADM": "Administration Block",
    "MTR-BLD-LIB": "Library",
    "MTR-BLD-HST": "Hostel A",
    "MTR-BLD-UTL": "Utility Block",
}


def calculate_billing(telemetry_df: pd.DataFrame) -> dict:
    """Calculate department chargebacks from sub-meter telemetry.

    Returns:
        dict with 'summary' DataFrame (per-dept totals) and
        'daily_trend' DataFrame (per-dept daily kWh).
    """
    tel = telemetry_df.copy()
    tel["timestamp"] = pd.to_datetime(tel["timestamp"], utc=True)
    tel["date"] = tel["timestamp"].dt.date
    tel["hour"] = tel["timestamp"].dt.hour   # compute BEFORE equip slice

    # Map assets to departments
    tel["department"] = tel["asset_id"].map(ASSET_DEPARTMENT_MAP)

    # Equipment-level telemetry only (not meters, to avoid double-counting)
    equip = tel[tel["reading_type"] == "equipment"].dropna(subset=["department"])


    # --- Summary by department ---
    summary = (
        equip.groupby("department")["interval_energy_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"interval_energy_kwh": "total_kwh"})
    )
    summary["cost_inr"] = summary["total_kwh"] * COST_PER_KWH_INR
    summary = summary.sort_values("total_kwh", ascending=False).reset_index(drop=True)

    # --- Daily trend per department ---
    daily = (
        equip.groupby(["date", "department"])["interval_energy_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"interval_energy_kwh": "daily_kwh"})
    )
    daily["date"] = daily["date"].astype(str)
    daily["cost_inr"] = daily["daily_kwh"] * COST_PER_KWH_INR

    # --- Building summary for the map view ---
    meters = tel[tel["reading_type"] == "meter"].copy()
    meters["building_name"] = meters["asset_id"].map(BUILDING_METER_MAP)
    building_summary = (
        meters.dropna(subset=["building_name"])
        .groupby("building_name")["interval_energy_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"interval_energy_kwh": "total_kwh"})
    )
    building_summary["cost_inr"] = building_summary["total_kwh"] * COST_PER_KWH_INR

    # --- AI Insight: flag highest after-hours consumer ---
    after_hours = equip[(equip["hour"] >= 19) | (equip["hour"] <= 6)]
    if not after_hours.empty:
        ah_dept = (
            after_hours.groupby("department")["interval_energy_kwh"]
            .sum()
            .sort_values(ascending=False)
        )
        top_dept = ah_dept.index[0]
        top_kwh = ah_dept.iloc[0]
        ai_insight = (
            f"🔍 **After-Hours Alert:** **{top_dept}** consumed "
            f"**{top_kwh:,.0f} kWh** after 7 PM / before 6 AM — "
            f"the highest after-hours draw on campus. "
            f"Recommend auditing equipment shut-off schedules for this department."
        )
    else:
        ai_insight = "No significant after-hours consumption detected."

    return {
        "summary": summary,
        "daily_trend": daily,
        "building_summary": building_summary,
        "ai_insight": ai_insight,
    }
