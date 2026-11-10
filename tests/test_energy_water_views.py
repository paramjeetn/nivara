"""Unit and integration tests for Energy and Water views (Step UX-7).
===================================================================
Tests:
1. Production hybrid forecaster (Seasonal-Naive + Ridge residual) bounds and accuracy.
2. Production forecaster with scenario levers (solar, LED, AC policy, occupancy).
3. Production forecaster with asset registry (per-meter vs campus total).
4. Water leak work order dispatch into SavingsLedger with plumbing tech assignment.
5. Water analytics and Minimum Night Flow (MNF) calculation integrity.
6. One value by default formatting compliance (no raw inline triplets).
7. Streamlit rendering smoke tests for render_energy and render_water.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from campus_energy.analytics.billing import calculate_billing
from campus_energy.analytics.forecasting import (
    compute_forecast_savings_range,
    generate_production_forecast,
)
from campus_energy.decision.ledger import SavingsLedger, WorkOrderItem
from campus_energy.decision.recommendation import Recommendation, SavingsBundle, SavingsRange
from campus_energy.simulator.water import (
    WATER_TARIFF_INR_PER_KL,
    analyze_water_data,
    generate_water_data,
)
from campus_energy.ui.components import format_inr, format_inr_compact
from campus_energy.ui.views.energy import render_energy
from campus_energy.ui.views.water import dispatch_water_leak_work_order, render_water


@pytest.fixture
def sample_assets_df() -> pd.DataFrame:
    """Fixture with meters, buildings, and sub-equipment."""
    return pd.DataFrame([
        {
            "asset_id": "MTR-BLD-A",
            "asset_name": "Academic Block A Main Incomer",
            "asset_type": "meter",
            "building_id": "BLD-A",
            "zone_id": "—",
            "rated_power_kw": 150.0,
        },
        {
            "asset_id": "MTR-BLD-B",
            "asset_name": "Academic Block B Main Incomer",
            "asset_type": "meter",
            "building_id": "BLD-B",
            "zone_id": "—",
            "rated_power_kw": 120.0,
        },
        {
            "asset_id": "EQ-A-CR101-AC",
            "asset_name": "Classroom 101 AC",
            "asset_type": "hvac",
            "building_id": "BLD-A",
            "zone_id": "ZONE-A1",
            "rated_power_kw": 10.0,
        },
    ])


@pytest.fixture
def synthetic_energy_telemetry() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 21 days of realistic 5-minute telemetry with weekly periodicity and weather."""
    start_dt = datetime(2026, 10, 1, 0, 0)
    periods = 288 * 21  # 21 days at 5-minute intervals
    timestamps = [start_dt + timedelta(minutes=5 * i) for i in range(periods)]

    rows_tel = []
    rows_ctx = []

    for t in timestamps:
        hr = t.hour
        dow = t.weekday()
        is_wknd = 1 if dow >= 5 else 0

        # Diurnal base pattern + weekly variation
        diurnal = 30.0 + 40.0 * np.sin(np.pi * (hr - 6) / 12) if 6 <= hr <= 20 else 15.0
        if is_wknd:
            diurnal *= 0.4

        temp = 28.0 + 8.0 * np.sin(np.pi * (hr - 9) / 12)
        occ = 100 if (9 <= hr <= 17 and not is_wknd) else 5

        # Meter A: 60% of campus load, Meter B: 40%
        p_a = max(2.0, diurnal * 0.60 + np.random.normal(0, 0.5))
        p_b = max(1.5, diurnal * 0.40 + np.random.normal(0, 0.4))

        rows_tel.append({"timestamp": t, "asset_id": "MTR-BLD-A", "power_kw": p_a})
        rows_tel.append({"timestamp": t, "asset_id": "MTR-BLD-B", "power_kw": p_b})
        rows_ctx.append({"timestamp": t, "outdoor_temperature_c": temp, "occupancy": occ, "holiday": False})

    return pd.DataFrame(rows_tel), pd.DataFrame(rows_ctx)


@pytest.fixture
def sample_facility_meta() -> Dict[str, Any]:
    """Sample facility metadata dictionary."""
    return {
        "facility_id": "bput_rourkela",
        "name": "Biju Patnaik University of Technology",
        "facility_type": "university",
        "contract_demand_kva": 450.0,
        "data_dir": "data/generated",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. PRODUCTION HYBRID FORECASTER TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_production_forecast_empty_inputs():
    """Verify safe fallback on empty DataFrames without crashing."""
    df = generate_production_forecast(pd.DataFrame(), pd.DataFrame())
    assert df.empty
    expected_cols = [
        "timestamp",
        "predicted_power_kw",
        "lower_bound",
        "upper_bound",
        "baseline_power_kw",
        "scenario_power_kw",
        "scenario_lower",
        "scenario_upper",
    ]
    for col in expected_cols:
        assert col in df.columns


def test_production_forecast_bounds_and_mape(synthetic_energy_telemetry, sample_assets_df):
    """Test that production forecast guarantees 0 <= lower <= predicted <= upper and achieves low MAPE."""
    tel_df, ctx_df = synthetic_energy_telemetry

    # Split last 24h as holdout to test MAPE
    latest_ts = tel_df["timestamp"].max()
    cutoff_ts = latest_ts - timedelta(days=1)

    train_tel = tel_df[tel_df["timestamp"] <= cutoff_ts].copy()
    test_tel = tel_df[tel_df["timestamp"] > cutoff_ts].copy()

    train_ctx = ctx_df[ctx_df["timestamp"] <= cutoff_ts].copy()
    test_ctx = ctx_df[ctx_df["timestamp"] > cutoff_ts].copy()

    # Generate campus 24h production forecast
    fcast = generate_production_forecast(
        telemetry_df=train_tel,
        context_df=train_ctx,
        assets_df=None,
        horizon_hours=24,
    )

    assert not fcast.empty
    assert len(fcast) == 288  # Exactly 24 hours at 5-minute intervals

    # Non-negative and monotonicity bounds test
    assert (fcast["predicted_power_kw"] >= 0).all()
    assert (fcast["lower_bound"] >= 0).all()
    assert (fcast["lower_bound"] <= fcast["predicted_power_kw"]).all()
    assert (fcast["predicted_power_kw"] <= fcast["upper_bound"]).all()
    assert (fcast["scenario_lower"] <= fcast["scenario_power_kw"]).all()
    assert (fcast["scenario_power_kw"] <= fcast["scenario_upper"]).all()

    # Calculate actual campus load in test period
    actual_campus = test_tel.groupby("timestamp")["power_kw"].sum().reset_index().sort_values("timestamp")
    actual_vals = actual_campus["power_kw"].values[:len(fcast)]
    pred_vals = fcast["predicted_power_kw"].values[:len(actual_vals)]

    valid_mask = actual_vals > 0.1
    mape = float(np.mean(np.abs((actual_vals[valid_mask] - pred_vals[valid_mask]) / actual_vals[valid_mask])))

    # Seasonal-naive + ridge hybrid should comfortably achieve < 20% MAPE on periodic data
    assert mape < 0.20, f"Expected MAPE < 20%, got {mape*100:.2f}%"


def test_production_forecast_scenario_levers(synthetic_energy_telemetry, sample_assets_df):
    """Verify that planning levers correctly reduce load and generate uncertainty intervals."""
    tel_df, ctx_df = synthetic_energy_telemetry

    # Baseline run
    base_fc = generate_production_forecast(
        telemetry_df=tel_df,
        context_df=ctx_df,
        assets_df=sample_assets_df,
        horizon_hours=24,
    )

    # Scenario run with active policy levers
    scen_fc = generate_production_forecast(
        telemetry_df=tel_df,
        context_df=ctx_df,
        assets_df=sample_assets_df,
        horizon_hours=24,
        solar_pv_kwp=100.0,
        ac_policy_reduction_pct=30.0,
        occupancy_reduction_pct=50.0,
        led_upgrade=True,
    )

    assert not scen_fc.empty
    # Scenario load should be strictly <= baseline forecast
    assert (scen_fc["scenario_power_kw"] <= scen_fc["predicted_power_kw"]).all()

    # Compute scenario savings bundle
    savings = compute_forecast_savings_range(scen_fc, cost_per_kwh=7.50, grid_emission_factor=0.82)

    assert "kwh" in savings
    assert "cost_inr" in savings
    assert "co2_kg" in savings

    # Monotonicity of uncertainty bounds: low <= expected <= high
    for metric in ["kwh", "cost_inr", "co2_kg"]:
        assert savings[metric]["low"] <= savings[metric]["expected"]
        assert savings[metric]["expected"] <= savings[metric]["high"]
        assert savings[metric]["expected"] > 0.0


def test_production_forecast_with_assets_per_meter(synthetic_energy_telemetry, sample_assets_df):
    """Verify per-meter forecast breakdown when assets_df is passed."""
    tel_df, ctx_df = synthetic_energy_telemetry

    per_meter_fc = generate_production_forecast(
        telemetry_df=tel_df,
        context_df=ctx_df,
        assets_df=sample_assets_df,
        horizon_hours=24,
        return_campus_total=False,
    )

    assert "asset_id" in per_meter_fc.columns
    unique_meters = per_meter_fc["asset_id"].unique().tolist()
    assert "MTR-BLD-A" in unique_meters
    assert "MTR-BLD-B" in unique_meters
    assert len(per_meter_fc) == 2 * 288  # 2 meters * 288 intervals


# ─────────────────────────────────────────────────────────────────────────────
# 2. WATER DESTINATION & WORK ORDER TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_dispatch_water_leak_work_order():
    """Verify dispatch_water_leak_work_order adds work order to ledger and assigns plumbing tech."""
    ledger = SavingsLedger()
    leak_record = {
        "leak_id": "LK-BLD-A-MNF",
        "building": "Academic Block A",
        "leak_type": "Overnight Pipe Leak (MNF > 3x Baseline)",
        "start_time": "2026-11-09 01:00",
        "duration": "16h",
        "duration_hours": 16,
        "leak_rate_lpm": 42.5,
        "lost_volume": "40.8 kL",
        "lost_volume_kl": 40.8,
        "cost_loss_inr": 1020.0,
        "severity": "High",
        "recommendation": "Inspect ground joint and basement PRV.",
    }

    wo_item = dispatch_water_leak_work_order(
        leak_record=leak_record,
        ledger=ledger,
        facility_id="test_fac",
        assignee="Plumbing Tech",
        sla_hours=12,
    )

    assert isinstance(wo_item, WorkOrderItem)
    assert wo_item.work_order_id == "WO-LK-BLD-A-MNF"
    assert wo_item.status == "work_order"
    assert wo_item.assignee == "Plumbing Tech"
    assert wo_item.priority == "high"
    assert wo_item.estimated_savings is not None
    assert wo_item.estimated_savings.inr.expected == 1020.0

    # Ensure idempotency / no duplicate inflation
    second_item = dispatch_water_leak_work_order(
        leak_record=leak_record,
        ledger=ledger,
        facility_id="test_fac",
    )
    assert second_item.work_order_id == wo_item.work_order_id
    assert len(ledger.get_items()) == 1


def test_water_analytics_kpi_and_diurnal():
    """Verify analyze_water_data outputs consistent AWWA M36 MNF diurnal curve and leak detection."""
    assets_df = pd.DataFrame([{"building_id": "BLD-A", "asset_id": "BLD-A", "asset_type": "building"}])
    water_df = generate_water_data(assets_df, days=14, seed=101)

    res = analyze_water_data(water_df)

    assert "kpis" in res
    assert "daily_trend" in res
    assert "diurnal_curve" in res
    assert "leak_events" in res

    kpis = res["kpis"]
    assert kpis["total_water_kl"] > 0
    assert kpis["water_cost_inr"] == pytest.approx(kpis["total_water_kl"] * WATER_TARIFF_INR_PER_KL, abs=1.0)
    assert kpis["cost_per_kl"] == WATER_TARIFF_INR_PER_KL

    diurnal = res["diurnal_curve"]
    assert len(diurnal) == 24
    # MNF window 01:00 to 05:00
    mnf_hours = diurnal[diurnal["hour"].isin([1, 2, 3, 4])]
    assert mnf_hours["is_mnf_zone"].all()


# ─────────────────────────────────────────────────────────────────────────────
# 3. ONE VALUE BY DEFAULT & FORMATTING TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_one_value_by_default_formatting():
    """Verify that numbers format as single values without raw inline range brackets."""
    assert format_inr(127500) == "₹1,27,500"
    assert "[" not in format_inr(127500)
    assert "–" not in format_inr(127500)

    assert format_inr_compact(1250000) == "₹12.5 Lakh"
    assert "[" not in format_inr_compact(1250000)

    assert format_inr(0) == "₹0"
    assert format_inr(750) == "₹750"


# ─────────────────────────────────────────────────────────────────────────────
# 4. STREAMLIT RENDERING SMOKE TESTS
# ─────────────────────────────────────────────────────────────────────────────

@patch("streamlit.plotly_chart")
@patch("streamlit.dataframe")
@patch("streamlit.metric")
@patch("streamlit.tabs")
def test_render_energy_smoke(mock_tabs, mock_metric, mock_df, mock_plotly, sample_facility_meta, sample_assets_df, synthetic_energy_telemetry):
    """Smoke test for render_energy ensuring all 5 tabs initialize without exception."""
    tel_df, ctx_df = synthetic_energy_telemetry
    ledger = SavingsLedger()

    # Create 5 dummy tab mock contexts
    mock_t1, mock_t2, mock_t3, mock_t4, mock_t5 = MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
    mock_tabs.return_value = [mock_t1, mock_t2, mock_t3, mock_t4, mock_t5]

    render_energy(
        facility_meta=sample_facility_meta,
        assets=sample_assets_df,
        tel=tel_df,
        ctx=ctx_df,
        events=pd.DataFrame(),
        complaints=pd.DataFrame(),
        ledger=ledger,
    )

    mock_tabs.assert_called_once()
    assert mock_metric.called


@patch("streamlit.plotly_chart")
@patch("streamlit.metric")
def test_render_water_smoke(mock_metric, mock_plotly, sample_facility_meta, sample_assets_df, synthetic_energy_telemetry):
    """Smoke test for render_water ensuring KPIs, charts, and leak alerts render cleanly."""
    tel_df, ctx_df = synthetic_energy_telemetry
    ledger = SavingsLedger()
    water_df = generate_water_data(sample_assets_df, days=7, seed=42)

    render_water(
        facility_meta=sample_facility_meta,
        assets=sample_assets_df,
        tel=tel_df,
        ctx=ctx_df,
        events=pd.DataFrame(),
        complaints=pd.DataFrame(),
        ledger=ledger,
        water_df=water_df,
    )

    assert mock_metric.called
