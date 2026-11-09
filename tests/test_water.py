"""Unit tests for Water Resource Module (TASK-07)."""

import pandas as pd
import pytest
from pathlib import Path

from campus_energy.simulator.water import (
    generate_water_data,
    analyze_water_data,
    WATER_TARIFF_INR_PER_KL,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "generated"


@pytest.fixture
def sample_assets_df():
    assets_path = DATA_DIR / "assets.csv"
    if assets_path.exists():
        return pd.read_csv(assets_path)
    return pd.DataFrame({
        "asset_id": ["BLD-A", "BLD-B"],
        "asset_type": ["building", "building"],
        "asset_name": ["Academic Block A", "Hostel Block B"],
        "building_id": ["BLD-A", "BLD-B"],
    })


def test_generate_water_data_schema(sample_assets_df):
    """Test output schema and row counts."""
    df = generate_water_data(sample_assets_df, days=7, seed=42)
    expected_cols = [
        "timestamp",
        "building_id",
        "water_meter_id",
        "flow_liters",
        "flow_rate_lpm",
        "is_anomaly",
        "anomaly_type",
    ]
    for col in expected_cols:
        assert col in df.columns

    bld_count = len(sample_assets_df[sample_assets_df["asset_type"] == "building"])
    if bld_count == 0:
        bld_count = sample_assets_df["building_id"].nunique()

    assert len(df) == bld_count * 7 * 24
    assert (df["flow_liters"] >= 0).all()
    assert (df["flow_rate_lpm"] >= 0).all()


def test_generate_water_data_leaks_injected(sample_assets_df):
    """Verify that both overnight and burst leaks are injected correctly."""
    df = generate_water_data(sample_assets_df, days=28, seed=42)

    # Check overnight pipe leak
    overnight = df[df["anomaly_type"] == "pipe_leak_overnight"]
    assert not overnight.empty
    assert (overnight["is_anomaly"] == True).all()
    ts = pd.to_datetime(overnight["timestamp"])
    assert ts.dt.hour.isin([1, 2, 3, 4]).all()

    # Check burst valve leak
    burst = df[df["anomaly_type"] == "burst_valve_leak"]
    assert not burst.empty
    assert (burst["is_anomaly"] == True).all()
    assert len(burst) == 6


def test_analyze_water_data(sample_assets_df):
    """Verify analytics output dictionary, KPI values, and chart data."""
    df = generate_water_data(sample_assets_df, days=28, seed=42)
    res = analyze_water_data(df)

    # Check dict structure
    assert "kpis" in res
    assert "daily_trend" in res
    assert "building_breakdown" in res
    assert "diurnal_curve" in res
    assert "leak_events" in res

    kpis = res["kpis"]
    assert kpis["total_water_kl"] > 0
    assert kpis["water_cost_inr"] > 0
    assert kpis["cost_per_kl"] == WATER_TARIFF_INR_PER_KL
    assert kpis["active_leaks"] >= 1
    assert kpis["avoidable_loss_kl"] > 0
    assert kpis["night_leak_lpm"] > 0

    # Daily trend
    daily = res["daily_trend"]
    assert len(daily) == 28
    assert "flow_kl" in daily.columns
    assert "rolling_avg_3d" in daily.columns

    # Building breakdown
    bld_breakdown = res["building_breakdown"]
    assert not bld_breakdown.empty
    assert "share_pct" in bld_breakdown.columns
    assert round(bld_breakdown["share_pct"].sum(), 0) == 100.0

    # Diurnal curve
    diurnal = res["diurnal_curve"]
    assert len(diurnal) == 24
    assert (diurnal["hour"] == list(range(24))).all()
    assert diurnal[diurnal["hour"].isin([1, 2, 3, 4])]["is_mnf_zone"].all()

    # Leak events
    leaks = res["leak_events"]
    assert not leaks.empty
    for required_field in ["building", "start_time", "duration", "lost_volume", "recommendation"]:
        assert required_field in leaks.columns


def test_analyze_empty_dataframe():
    """Verify analyze_water_data handles empty input gracefully without errors."""
    res = analyze_water_data(pd.DataFrame())
    assert res["kpis"]["total_water_kl"] == 0.0
    assert res["kpis"]["active_leaks"] == 0
    assert res["daily_trend"].empty
    assert res["leak_events"].empty


def test_generate_water_data_fallback_building():
    """Verify fallback when assets dataframe only has building_id or is minimal."""
    min_assets = pd.DataFrame({"building_id": ["BLD-OFFICE-1"]})
    df = generate_water_data(min_assets, days=2, seed=123)
    assert len(df) == 1 * 2 * 24
    assert df["building_id"].iloc[0] == "BLD-OFFICE-1"
    assert df["water_meter_id"].iloc[0] == "WM-BLD-OFFICE-1"
