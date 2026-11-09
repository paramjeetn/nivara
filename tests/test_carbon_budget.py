"""Unit tests for the Carbon Budget Tracker module."""
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from campus_energy.analytics.carbon_budget import calculate_carbon_budget


@pytest.fixture
def synthetic_telemetry():
    """Create 20 days of synthetic meter telemetry."""
    dates = pd.date_range("2025-09-01", periods=20, freq="D")
    records = []
    for d in dates:
        # 1000 kWh per day -> ~0.82 t CO2 per day
        records.append({
            "timestamp": d.strftime("%Y-%m-%d 12:00:00"),
            "reading_type": "meter",
            "interval_energy_kwh": 1000.0,
            "asset_id": "MTR-MAIN",
        })
    return pd.DataFrame(records)


def test_calculate_carbon_budget_keys_and_shapes(synthetic_telemetry):
    """Test returned keys, dataframe columns, and dimensions."""
    res = calculate_carbon_budget(
        synthetic_telemetry,
        floor_area_m2=10000.0,
        facility_type="university",
    )

    assert "summary" in res
    assert "burn_down_df" in res
    assert "projection_df" in res

    summary = res["summary"]
    required_keys = {
        "monthly_budget_tco2",
        "current_cumulative_tco2",
        "projected_month_end_tco2",
        "remaining_budget_tco2",
        "overshoot_tco2",
        "overshoot_pct",
        "burn_rate_tco2_day",
        "days_observed",
        "days_remaining",
        "daily_curtail_kwh",
        "status",
        "alert",
        "consumption_pct",
        "facility_type",
        "floor_area_m2",
    }
    assert required_keys.issubset(summary.keys())

    burn_down_df = res["burn_down_df"]
    assert len(burn_down_df) == 20
    assert "cumulative_tco2" in burn_down_df.columns
    assert "ideal_target_tco2" in burn_down_df.columns
    assert "budget_ceiling_tco2" in burn_down_df.columns
    assert burn_down_df["day_index"].iloc[0] == 1
    assert burn_down_df["day_index"].iloc[-1] == 20

    projection_df = res["projection_df"]
    # Days 20 to 30 inclusive = 11 rows
    assert len(projection_df) == 11
    assert projection_df["day_index"].iloc[0] == 20
    assert projection_df["day_index"].iloc[-1] == 30
    assert "projected_cumulative_tco2" in projection_df.columns


def test_facility_types_automatic_budget():
    """Verify automatic budget calculation based on ECBC baselines and NAPCC 20% target."""
    dummy_df = pd.DataFrame([{
        "timestamp": "2025-09-01 12:00:00",
        "reading_type": "meter",
        "interval_energy_kwh": 500.0,
    }])

    # floor_area = 10,000 m²
    # annual_target = baseline * 0.8 * 10,000
    # monthly_budget = round((annual_target / 12 * 0.82) / 1000, 1)

    # University: baseline 100 -> 54.7 t CO2
    res_uni = calculate_carbon_budget(dummy_df, floor_area_m2=10000.0, facility_type="university")
    assert res_uni["summary"]["monthly_budget_tco2"] == 54.7

    # Hospital: baseline 300 -> 164.0 t CO2
    res_hosp = calculate_carbon_budget(dummy_df, floor_area_m2=10000.0, facility_type="hospital")
    assert res_hosp["summary"]["monthly_budget_tco2"] == 164.0

    # Government Office: baseline 140 -> 76.5 t CO2
    res_govt = calculate_carbon_budget(dummy_df, floor_area_m2=10000.0, facility_type="govt_office")
    assert res_govt["summary"]["monthly_budget_tco2"] == 76.5

    # Industrial: baseline 450 -> 246.0 t CO2
    res_ind = calculate_carbon_budget(dummy_df, floor_area_m2=10000.0, facility_type="industrial")
    assert res_ind["summary"]["monthly_budget_tco2"] == 246.0


def test_status_and_overshoot_logic(synthetic_telemetry):
    """Test ON_TRACK, WARNING, and CRITICAL thresholds."""
    # synthetic_telemetry has 20 days with 1,000 kWh/day = 0.82 t CO2/day.
    # Cumulative over 20 days = 16.4 t CO2.
    # Projected over 30 days = 24.6 t CO2.

    # 1. ON_TRACK: Budget is 30.0 t CO2 (> 24.6)
    res_on_track = calculate_carbon_budget(
        synthetic_telemetry,
        monthly_budget_tco2=30.0,
    )
    assert res_on_track["summary"]["status"] == "ON_TRACK"
    assert "On Track" in res_on_track["summary"]["alert"]
    assert res_on_track["summary"]["overshoot_tco2"] == 0.0
    assert res_on_track["summary"]["daily_curtail_kwh"] == 0.0

    # 2. WARNING: Budget is 23.5 t CO2.
    # Projected is 24.6 -> overshoot = 1.1 t CO2, pct = (1.1 / 23.5) * 100 = 4.7% (<= 10%)
    res_warning = calculate_carbon_budget(
        synthetic_telemetry,
        monthly_budget_tco2=23.5,
    )
    assert res_warning["summary"]["status"] == "WARNING"
    assert "Budget Risk" in res_warning["summary"]["alert"]
    assert res_warning["summary"]["overshoot_tco2"] > 0
    assert res_warning["summary"]["daily_curtail_kwh"] > 0

    # 3. CRITICAL: Budget is 18.0 t CO2.
    # Projected is 24.6 -> overshoot = 6.6 t CO2, pct = (6.6 / 18.0) * 100 = 36.7% (> 10%)
    res_crit = calculate_carbon_budget(
        synthetic_telemetry,
        monthly_budget_tco2=18.0,
    )
    assert res_crit["summary"]["status"] == "CRITICAL"
    assert "Critical Overshoot" in res_crit["summary"]["alert"]
    assert res_crit["summary"]["overshoot_pct"] > 10.0
    assert res_crit["summary"]["daily_curtail_kwh"] > 0


def test_custom_budget_override(synthetic_telemetry):
    """Test explicitly setting a custom monthly budget."""
    res = calculate_carbon_budget(synthetic_telemetry, monthly_budget_tco2=50.0)
    assert res["summary"]["monthly_budget_tco2"] == 50.0


def test_empty_telemetry():
    """Test handling of empty telemetry dataframe."""
    empty_df = pd.DataFrame()
    res = calculate_carbon_budget(empty_df, floor_area_m2=10000.0)
    assert res["summary"]["current_cumulative_tco2"] == 0.0
    assert res["summary"]["projected_month_end_tco2"] == 0.0
    assert res["summary"]["status"] == "ON_TRACK"
    assert len(res["burn_down_df"]) == 0


def test_with_real_facility_telemetry():
    """Test against actual generated telemetry file if available."""
    real_path = Path("data/generated/telemetry.csv")
    if not real_path.exists():
        pytest.skip("Generated telemetry not found.")

    tel = pd.read_csv(real_path)
    res = calculate_carbon_budget(tel, floor_area_m2=12000.0, facility_type="university")

    assert res["summary"]["days_observed"] >= 28
    assert res["summary"]["current_cumulative_tco2"] > 0
    assert res["summary"]["projected_month_end_tco2"] > 0
    assert len(res["burn_down_df"]) == res["summary"]["days_observed"]
