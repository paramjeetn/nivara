"""Comprehensive unit and end-to-end tests for the Savings Ledger & M&V Verification Engine (Step 10)."""
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from campus_energy import assumptions
from campus_energy.decision.ledger import LedgerSummary, SavingsLedger, WorkOrderItem
from campus_energy.decision.recommendation import Recommendation, SavingsBundle, SavingsRange
from campus_energy.decision.tariffs import IndianTariff
from campus_energy.decision.engine import generate_recommendations_from_alerts
from campus_energy.schemas import Alert, AlertStatus, EventType, Severity
from campus_energy.simulator.events import inject_events, simulate_fault_resolution


@pytest.fixture
def sample_recommendation():
    """Create a standardized actionable recommendation for test verification."""
    savings = SavingsBundle.from_expected(
        kwh_expected=100.0,
        inr_per_kwh=8.50,
        co2_per_kwh=0.82,
        spread=0.20,
    )
    return Recommendation(
        id="REC-2026-001",
        facility_id="university",
        type="anomaly_fix",
        target="EQ-HVAC-01",
        action_text="Reset chiller overnight timer schedule setback.",
        trigger="hvac_left_on alert detected after operating hours",
        savings=savings,
        cost=0.0,
        payback_months=0.0,
        assumption_refs=["tariff.electricity_commercial", "emissions.grid_factor"],
        owner_role="operations",
        lifecycle="proposed",
        confidence=0.92,
        is_estimate=True,
    )


@pytest.fixture
def sample_telemetry_pair():
    """Create synthetic post-completion telemetry and baseline counterfactual DataFrames."""
    start_dt = datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc)
    completion_dt = datetime(2026, 11, 1, 20, 0, tzinfo=timezone.utc)
    end_dt = datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc)

    # 5-minute intervals from 18:00 to 23:00 (5 hours * 12 = 60 intervals)
    timestamps = pd.date_range(start_dt, end_dt, freq="5min")

    # Baseline counterfactual: chiller remained stuck ON at 24 kW
    base_rows = []
    # Post-resolution telemetry: chiller stuck ON at 24 kW until 20:00, then reset to 4 kW
    post_rows = []

    for ts in timestamps:
        # Counterfactual: 24 kW constantly
        base_rows.append({
            "reading_id": f"RDG-BASE-{ts.strftime('%H%M')}",
            "timestamp": ts.isoformat(),
            "asset_id": "EQ-HVAC-01",
            "power_kw": 24.0,
            "interval_energy_kwh": 24.0 / 12.0,  # 2.0 kWh per 5 min
            "state": "ON",
        })

        # Actual post-telemetry: drops to 4 kW at and after 20:00
        p_act = 24.0 if ts < completion_dt else 4.0
        post_rows.append({
            "reading_id": f"RDG-POST-{ts.strftime('%H%M')}",
            "timestamp": ts.isoformat(),
            "asset_id": "EQ-HVAC-01",
            "power_kw": p_act,
            "interval_energy_kwh": p_act / 12.0,
            "state": "ON" if ts < completion_dt else "OFF",
        })

    return completion_dt, pd.DataFrame(post_rows), pd.DataFrame(base_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Lifecycle State Machine Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ledger_full_lifecycle_transitions(sample_recommendation, sample_telemetry_pair):
    """Test full closed-loop state transitions: proposed -> accepted -> work_order -> done -> verified."""
    completion_dt, post_df, base_df = sample_telemetry_pair
    ledger = SavingsLedger()

    # 1. Proposed
    item = ledger.add_recommendation(sample_recommendation)
    assert item.status == "proposed"
    assert item.recommendation.lifecycle == "proposed"
    assert item.work_order_id == "WO-2026-001"
    assert item.assignee == "HVAC Lead"
    assert item.priority in ["critical", "high", "medium", "low"]
    assert item.verified_savings is None
    assert not item.is_verified

    # 2. Accepted
    accepted_item = ledger.accept_recommendation(item.work_order_id)
    assert accepted_item.status == "accepted"
    assert accepted_item.recommendation.lifecycle == "accepted"

    # 3. Dispatched as Work Order
    wo_item = ledger.create_work_order(
        rec_id=accepted_item.work_order_id,
        assignee="HVAC Lead Engineer",
        sla_hours=12,
        priority="critical",
    )
    assert wo_item.status == "work_order"
    assert wo_item.recommendation.lifecycle == "work_order"
    assert wo_item.assignee == "HVAC Lead Engineer"
    assert wo_item.priority == "critical"
    assert wo_item.sla_deadline > wo_item.created_at

    # 4. Done (physical work finished)
    done_item = ledger.mark_done(wo_item.work_order_id, completed_at=completion_dt)
    assert done_item.status == "done"
    assert done_item.recommendation.lifecycle == "done"
    assert done_item.completed_at == completion_dt

    # 5. Verified via M&V Engine
    verified_item = ledger.complete_and_verify(
        wo_id=done_item.work_order_id,
        completion_time=completion_dt,
        post_telemetry_df=post_df,
        baseline_df=base_df,
    )
    assert verified_item.status == "verified"
    assert verified_item.recommendation.lifecycle == "verified"
    assert verified_item.is_verified
    assert verified_item.verified_savings is not None
    assert verified_item.verified_savings.kwh.expected > 0.0
    assert verified_item.verified_savings.inr.expected > 0.0
    assert verified_item.verified_savings.co2_kg.expected > 0.0
    assert "IPMVP Option B" in verified_item.verification_notes


# ─────────────────────────────────────────────────────────────────────────────
# 2. Counterfactual M&V Calculation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_counterfactual_mv_verification_math(sample_recommendation, sample_telemetry_pair):
    """Test IPMVP Option B counterfactual verification calculation aligns with physics."""
    completion_dt, post_df, base_df = sample_telemetry_pair
    ledger = SavingsLedger()
    item = ledger.add_recommendation(sample_recommendation)

    custom_tariff = IndianTariff(energy_rate_inr_per_kwh=10.0)

    # Post-completion period: 20:00 to 23:00 (3 hours = 36 intervals)
    # Counterfactual power: 24 kW => 24 kW * 3 h = 72 kWh
    # Actual post power: 4 kW => 4 kW * 3 h = 12 kWh
    # ΔkWh = 72 - 12 = 60 kWh
    verified_item = ledger.complete_and_verify(
        wo_id=item.work_order_id,
        completion_time=completion_dt,
        post_telemetry_df=post_df,
        baseline_df=base_df,
        tariff=custom_tariff,
    )

    kwh_exp = verified_item.verified_savings.kwh.expected
    inr_exp = verified_item.verified_savings.inr.expected
    co2_exp = verified_item.verified_savings.co2_kg.expected

    # Energy: ~60 kWh (within floating-point discretization tolerance)
    assert abs(kwh_exp - 60.0) <= 2.0
    # Cost: 60 kWh * ₹10.0/kWh = ₹600.0
    assert abs(inr_exp - (kwh_exp * 10.0)) <= 0.5
    # GHG: 60 kWh * 0.82 kg/kWh = 49.2 kg CO2e
    assert abs(co2_exp - (kwh_exp * 0.82)) <= 0.5

    # Range monotonicity on verified bundle
    assert verified_item.verified_savings.kwh.low <= kwh_exp <= verified_item.verified_savings.kwh.high
    assert verified_item.verified_savings.inr.low <= inr_exp <= verified_item.verified_savings.inr.high
    assert verified_item.verified_savings.co2_kg.low <= co2_exp <= verified_item.verified_savings.co2_kg.high


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hit-Rate & Ledger Summary Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ledger_summary_and_hit_rate(sample_recommendation):
    """Test hit rate calculation when verified falls within vs outside estimated range."""
    ledger = SavingsLedger()

    # Rec 1: Expected 100 kWh -> INR Expected ₹850 [Low ₹680, High ₹1020]
    rec1 = sample_recommendation
    it1 = ledger.add_recommendation(rec1)

    # Simulate verification with 100 kWh verified (= ₹850 INR) -> HIT
    it1.status = "verified"
    it1.verified_savings = SavingsBundle.from_expected(100.0, inr_per_kwh=8.50, spread=0.05)

    summary1 = ledger.get_summary()
    assert summary1.total_actions == 1
    assert summary1.verified_count == 1
    assert summary1.cumulative_verified_inr == 850.0
    assert summary1.estimate_hit_rate_pct == 100.0

    # Rec 2: Expected 100 kWh -> INR Expected ₹850 [Low ₹680, High ₹1020]
    rec2 = sample_recommendation.model_copy(update={"id": "REC-2026-002", "target": "EQ-HVAC-02"})
    it2 = ledger.add_recommendation(rec2)

    # Simulate verification with 20 kWh verified (= ₹170 INR < ₹680) -> MISS / Underperformed
    it2.status = "verified"
    it2.verified_savings = SavingsBundle.from_expected(20.0, inr_per_kwh=8.50, spread=0.05)

    summary2 = ledger.get_summary()
    assert summary2.total_actions == 2
    assert summary2.verified_count == 2
    assert summary2.cumulative_verified_inr == 850.0 + 170.0
    # 1 hit out of 2 = 50.0%
    assert summary2.estimate_hit_rate_pct == 50.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-End Simulation, Resolution, and Verification Loop
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_fault_resolution_and_verification():
    """End-to-end integration test:

    1. Generate baseline telemetry.
    2. Inject anomaly event (lights left on).
    3. Generate alert and decision recommendation.
    4. Log recommendation into SavingsLedger and dispatch Work Order.
    5. Simulate physical fault resolution via `simulate_fault_resolution`.
    6. Complete and verify work order using counterfactual M&V.
    7. Verify ledger summary captures positive avoided cost and valid hit rate.
    """
    # 1. Base telemetry & assets
    base_ts = pd.date_range("2026-11-01 12:00:00+05:30", periods=72, freq="5min")
    asset_id = "EQ-LT-101"
    rated_kw = 10.0

    # Normal baseline: lights off (0 kW)
    base_rows = []
    for ts in base_ts:
        base_rows.append({
            "reading_id": f"RDG-{ts.strftime('%H%M%S')}",
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            "asset_id": asset_id,
            "reading_type": "equipment",
            "power_kw": 0.0,
            "interval_energy_kwh": 0.0,
            "state": "OFF",
        })
    normal_baseline_df = pd.DataFrame(base_rows)

    # 2. Inject fault: lights left on from 13:00 to 17:00 (4 hours)
    tz_ist = timezone(timedelta(hours=5, minutes=30))
    start_time = datetime(2026, 11, 1, 13, 0, tzinfo=tz_ist)
    end_time = datetime(2026, 11, 1, 17, 0, tzinfo=tz_ist)

    faulty_df = normal_baseline_df.copy()
    faulty_df["_ts_utc"] = pd.to_datetime(faulty_df["timestamp"]).dt.tz_convert(timezone.utc)

    fault_mask = (faulty_df["_ts_utc"] >= start_time) & (faulty_df["_ts_utc"] <= end_time)
    faulty_df.loc[fault_mask, "power_kw"] = rated_kw
    faulty_df.loc[fault_mask, "interval_energy_kwh"] = rated_kw / 12.0
    faulty_df.loc[fault_mask, "state"] = "ON"
    faulty_df.drop(columns=["_ts_utc"], inplace=True)

    # 3. Create Alert & Recommendation
    alert = Alert(
        alert_id="ALT-2026-001",
        detected_at=datetime(2026, 11, 1, 14, 0, tzinfo=tz_ist),
        event_start_estimate=start_time,
        asset_id=asset_id,
        building_id="BLD-LIB",
        zone_id="ZONE-L1",
        anomaly_type=EventType.LIGHTS_LEFT_ON.value,
        severity=Severity.HIGH,
        actual_value=10.0,
        expected_value=0.0,
        deviation=10.0,
        evidence=f"Lighting asset {asset_id} drawing {rated_kw} kW after closing.",
        confidence=0.95,
        recommendation=f"Switch off lighting circuits on {asset_id} and inspect relay.",
        status=AlertStatus.ACTIVE,
    )

    recs = generate_recommendations_from_alerts([alert])
    assert len(recs) >= 1
    rec = recs[0]

    # 4. Ingest into SavingsLedger & Dispatch Work Order
    ledger = SavingsLedger()
    item = ledger.add_recommendation(rec)
    wo = ledger.create_work_order(item.work_order_id, assignee="Electrical Maintenance Crew")
    assert wo.status == "work_order"

    # 5. Simulate resolution fix at 15:00 (midway through event)
    resolution_time = datetime(2026, 11, 1, 15, 0, tzinfo=tz_ist)
    event_dict = {
        "asset_id": asset_id,
        "start_time": start_time,
        "end_time": end_time,
    }

    resolved_telemetry = simulate_fault_resolution(
        telemetry_df=faulty_df,
        event=event_dict,
        resolution_time=resolution_time,
        baseline_df=normal_baseline_df,
    )

    # Verify that post-resolution, power returned to baseline (0 kW)
    res_ts_utc = pd.to_datetime(resolved_telemetry["timestamp"]).dt.tz_convert(timezone.utc)
    post_res_mask = (resolved_telemetry["asset_id"] == asset_id) & (res_ts_utc >= resolution_time) & (res_ts_utc <= end_time)
    assert (resolved_telemetry.loc[post_res_mask, "power_kw"] == 0.0).all()

    # 6. Complete and Verify via M&V Engine
    # Here, faulty_df serves as the counterfactual baseline (what would have occurred had the fault persisted)
    verified_wo = ledger.complete_and_verify(
        wo_id=wo.work_order_id,
        completion_time=resolution_time,
        post_telemetry_df=resolved_telemetry,
        baseline_df=faulty_df,
    )

    assert verified_wo.status == "verified"
    assert verified_wo.verified_savings is not None

    # Physics check: from 15:00 to 17:00 is 2 hours.
    # At 10 kW, 2 hours = 20 kWh saved!
    kwh_verified = verified_wo.verified_savings.kwh.expected
    assert abs(kwh_verified - 20.0) <= 1.0

    inr_verified = verified_wo.verified_savings.inr.expected
    assert inr_verified > 0.0

    # 7. Check Ledger Summary
    summary = ledger.get_summary()
    assert summary.total_actions == 1
    assert summary.verified_count == 1
    assert summary.cumulative_verified_kwh == kwh_verified
    assert summary.cumulative_verified_inr == inr_verified
    assert summary.cumulative_verified_co2_kg > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Queries, Filters, and Simulator Resolution Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

def test_ledger_queries_and_filters(sample_recommendation):
    """Test get_items filtering by status and get_item lookup by ID."""
    ledger = SavingsLedger()
    it1 = ledger.add_recommendation(sample_recommendation)
    rec2 = sample_recommendation.model_copy(update={"id": "REC-2026-002", "target": "EQ-HVAC-02"})
    it2 = ledger.add_recommendation(rec2)

    assert len(ledger.get_items()) == 2
    assert len(ledger.get_items("proposed")) == 2
    assert len(ledger.get_items("work_order")) == 0

    ledger.create_work_order(it1.work_order_id)
    assert len(ledger.get_items("proposed")) == 1
    assert len(ledger.get_items("work_order")) == 1

    # Lookup by WO ID and by Rec ID
    assert ledger.get_item(it1.work_order_id) is not None
    assert ledger.get_item(sample_recommendation.id) is not None
    assert ledger.get_item("NON-EXISTENT") is None


def test_ledger_error_handling(sample_recommendation):
    """Test exception raising for nonexistent recommendations."""
    ledger = SavingsLedger()
    with pytest.raises(KeyError):
        ledger.accept_recommendation("REC-DOES-NOT-EXIST")
    with pytest.raises(KeyError):
        ledger.create_work_order("REC-DOES-NOT-EXIST")
    with pytest.raises(KeyError):
        ledger.mark_done("REC-DOES-NOT-EXIST")


def test_ledger_summary_zero_verified():
    """Test that empty or unverified ledger returns 0.0 hit rate and zeros for verified metrics."""
    ledger = SavingsLedger()
    summary = ledger.get_summary()
    assert summary.total_actions == 0
    assert summary.verified_count == 0
    assert summary.cumulative_estimated_inr == 0.0
    assert summary.cumulative_verified_inr == 0.0
    assert summary.estimate_hit_rate_pct == 0.0


def test_simulate_fault_resolution_formats():
    """Test simulate_fault_resolution accepting dict, Series, or string asset_id."""
    base_ts = pd.date_range("2026-11-01 10:00:00+05:30", periods=12, freq="5min")
    df_base = pd.DataFrame([{
        "reading_id": f"R-{i}",
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
        "asset_id": "EQ-PUMP-01",
        "power_kw": 5.0,
        "interval_energy_kwh": 5.0 / 12.0,
        "flow_lpm": 120.0,
    } for i, ts in enumerate(base_ts)])

    # Inject elevated power and zero flow
    df_faulty = df_base.copy()
    df_faulty["power_kw"] = 15.0
    df_faulty["flow_lpm"] = 0.0

    # Resolve from midpoint
    res_time = base_ts[6]

    # Test with string asset_id
    res_df = simulate_fault_resolution(
        telemetry_df=df_faulty,
        event="EQ-PUMP-01",
        resolution_time=res_time,
        baseline_df=df_base,
    )

    # Post-resolution rows should have power restored to 5.0 and flow to 120.0
    ts_parsed = pd.to_datetime(res_df["timestamp"])
    post_mask = ts_parsed >= res_time
    assert (res_df.loc[post_mask, "power_kw"] == 5.0).all()
    assert (res_df.loc[post_mask, "flow_lpm"] == 120.0).all()

    # Pre-resolution rows should remain faulty
    pre_mask = ts_parsed < res_time
    assert (res_df.loc[pre_mask, "power_kw"] == 15.0).all()
    assert (res_df.loc[pre_mask, "flow_lpm"] == 0.0).all()
