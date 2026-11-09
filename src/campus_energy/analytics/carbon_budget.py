"""Carbon Budget Tracker (Monthly CO2 Budget linked to NAPCC / Net-Zero Target).

Calculates cumulative carbon consumption, pro-rated daily allowances,
projected month-end emissions, and overshoot risk with load curtailment guidance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from campus_energy import assumptions
from campus_energy.analytics.esg import BEE_EPI_BENCHMARKS


def calculate_carbon_budget(
    telemetry_df: pd.DataFrame,
    floor_area_m2: float = 10000.0,
    facility_type: str = "university",
    monthly_budget_tco2: float | None = None,
) -> dict:
    """Calculate monthly carbon budget metrics, daily burn-down, and 30-day projection.

    Args:
        telemetry_df: Telemetry dataframe with energy readings.
        floor_area_m2: Conditioned floor area of the facility in m².
        facility_type: One of 'university', 'hospital', 'govt_office', 'industrial'.
        monthly_budget_tco2: Optional explicit monthly carbon budget in tonnes CO₂.
            If None, derived from NAPCC 20% reduction target vs ECBC baseline.

    Returns:
        dict with keys:
            - summary: dict containing budget, cumulative emissions, projection, overshoot & status
            - burn_down_df: DataFrame with daily cumulative emissions up to observed day
            - projection_df: DataFrame with projected trajectory up to day 30
    """
    # 1. Determine monthly budget
    grid_emission_factor: float = assumptions.get_value("emissions.grid_factor")
    napcc_pct: float = assumptions.get_value("policy.napcc_reduction_target_pct")
    ceiling_days: float = assumptions.get_value("budget.month_ceiling_days")

    if monthly_budget_tco2 is None:
        ecbc_baseline = BEE_EPI_BENCHMARKS.get(
            facility_type, BEE_EPI_BENCHMARKS["university"]
        )["ecbc_baseline"]
        annual_target_kwh = ecbc_baseline * (1.0 - napcc_pct / 100.0) * floor_area_m2
        monthly_budget_tco2 = round(
            (annual_target_kwh / 12.0 * grid_emission_factor) / 1000.0, 1
        )
        monthly_budget_tco2 = max(1.0, monthly_budget_tco2)
    else:
        monthly_budget_tco2 = max(1.0, round(float(monthly_budget_tco2), 2))

    # 2. Extract meter energy readings
    tel = telemetry_df.copy()
    if "reading_type" in tel.columns and (tel["reading_type"] == "meter").any():
        meters = tel[tel["reading_type"] == "meter"].copy()
    else:
        meters = tel.copy()

    if meters.empty or "interval_energy_kwh" not in meters.columns:
        # Graceful empty handling
        summary = {
            "monthly_budget_tco2": monthly_budget_tco2,
            "current_cumulative_tco2": 0.0,
            "projected_month_end_tco2": 0.0,
            "remaining_budget_tco2": monthly_budget_tco2,
            "overshoot_tco2": 0.0,
            "overshoot_pct": -100.0,
            "burn_rate_tco2_day": 0.0,
            "days_observed": 0,
            "days_remaining": int(ceiling_days),
            "daily_curtail_kwh": 0.0,
            "status": "ON_TRACK",
            "alert": "✅ On Track: Facility is projected to stay within the monthly NAPCC carbon budget ceiling.",
            "consumption_pct": 0.0,
            "facility_type": facility_type,
            "floor_area_m2": floor_area_m2,
        }
        burn_down_df = pd.DataFrame(
            columns=[
                "date",
                "day_index",
                "daily_kwh",
                "daily_tco2",
                "cumulative_tco2",
                "ideal_target_tco2",
                "budget_ceiling_tco2",
            ]
        )
        projection_df = pd.DataFrame(
            columns=[
                "date",
                "day_index",
                "projected_cumulative_tco2",
                "ideal_target_tco2",
                "budget_ceiling_tco2",
            ]
        )
        return {
            "summary": summary,
            "burn_down_df": burn_down_df,
            "projection_df": projection_df,
        }

    # Ensure datetime & date column
    meters["timestamp"] = pd.to_datetime(meters["timestamp"])
    meters["date"] = meters["timestamp"].dt.date

    # Daily aggregation
    daily = (
        meters.groupby("date")["interval_energy_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"interval_energy_kwh": "daily_kwh"})
    )
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["date"] = daily["date"].astype(str)

    # Day index 1..N
    daily["day_index"] = np.arange(1, len(daily) + 1)
    daily["daily_tco2"] = round(
        daily["daily_kwh"] * grid_emission_factor / 1000.0, 3
    )
    daily["cumulative_tco2"] = round(daily["daily_tco2"].cumsum(), 3)
    daily["ideal_target_tco2"] = round(
        (daily["day_index"] / ceiling_days) * monthly_budget_tco2, 3
    )
    daily["budget_ceiling_tco2"] = monthly_budget_tco2

    d_obs = len(daily)
    current_cumulative_tco2 = (
        float(daily["cumulative_tco2"].iloc[-1]) if d_obs > 0 else 0.0
    )
    burn_rate_tco2_day = current_cumulative_tco2 / max(d_obs, 1)

    projected_month_end_tco2 = round(burn_rate_tco2_day * ceiling_days, 2)
    remaining_budget_tco2 = round(monthly_budget_tco2 - current_cumulative_tco2, 2)
    overshoot_tco2 = max(0.0, round(projected_month_end_tco2 - monthly_budget_tco2, 2))
    overshoot_pct = round(
        ((projected_month_end_tco2 - monthly_budget_tco2) / monthly_budget_tco2) * 100.0, 1
    )
    days_remaining = max(0, int(ceiling_days) - d_obs)

    # Curtailment calculation: required daily kWh reduction over remaining days to avoid overshoot
    if overshoot_tco2 > 0 and days_remaining > 0:
        excess_kwh = (overshoot_tco2 * 1000.0) / grid_emission_factor
        daily_curtail_kwh = round(excess_kwh / days_remaining, 1)
    else:
        daily_curtail_kwh = 0.0

    # Determine status & alert message
    if projected_month_end_tco2 <= monthly_budget_tco2:
        status = "ON_TRACK"
        alert = "✅ On Track: Facility is projected to stay within the monthly NAPCC carbon budget ceiling."
    elif overshoot_pct <= 10.0:
        status = "WARNING"
        alert = (
            f"⚠️ Budget Risk: Projected to exceed allowance by {overshoot_tco2} t CO₂ ({overshoot_pct}%). "
            f"Curtailed load of {daily_curtail_kwh:.0f} kWh/day recommended."
        )
    else:
        status = "CRITICAL"
        alert = (
            f"🚨 Critical Overshoot: Projected to breach carbon ceiling by {overshoot_tco2} t CO₂ (+{overshoot_pct}%). "
            f"Immediate dispatch and HVAC curtailment required."
        )

    consumption_pct = round((current_cumulative_tco2 / monthly_budget_tco2) * 100.0, 1)

    summary = {
        "monthly_budget_tco2": monthly_budget_tco2,
        "current_cumulative_tco2": round(current_cumulative_tco2, 2),
        "projected_month_end_tco2": projected_month_end_tco2,
        "remaining_budget_tco2": remaining_budget_tco2,
        "overshoot_tco2": overshoot_tco2,
        "overshoot_pct": overshoot_pct,
        "burn_rate_tco2_day": round(burn_rate_tco2_day, 3),
        "days_observed": int(d_obs),
        "days_remaining": int(days_remaining),
        "daily_curtail_kwh": daily_curtail_kwh,
        "status": status,
        "alert": alert,
        "consumption_pct": consumption_pct,
        "facility_type": facility_type,
        "floor_area_m2": floor_area_m2,
    }

    # Build projection DataFrame starting from day d_obs up to month ceiling day
    last_date = pd.to_datetime(daily["date"].iloc[-1])
    proj_rows = []
    start_day = d_obs if d_obs > 0 else 1
    for day_idx in range(start_day, int(ceiling_days) + 1):
        delta_days = day_idx - d_obs
        p_date = (last_date + pd.Timedelta(days=int(delta_days))).strftime("%Y-%m-%d")
        p_val = round(current_cumulative_tco2 + max(0, delta_days) * burn_rate_tco2_day, 3)
        ideal_val = round((day_idx / ceiling_days) * monthly_budget_tco2, 3)
        proj_rows.append(
            {
                "date": p_date,
                "day_index": day_idx,
                "projected_cumulative_tco2": p_val,
                "ideal_target_tco2": ideal_val,
                "budget_ceiling_tco2": monthly_budget_tco2,
            }
        )
    projection_df = pd.DataFrame(proj_rows)

    return {
        "summary": summary,
        "burn_down_df": daily,
        "projection_df": projection_df,
    }
