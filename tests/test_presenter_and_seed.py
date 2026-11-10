"""Unit and Integration Tests for Step UX-10: Presenter Kit and Realistic Seed State.

Validates:
1. `seed_operational_history` populates SavingsLedger with:
   - 3 completed & verified work orders (WO-2025-089, WO-2025-094, WO-2025-098)
   - Verified savings of ₹4,800/mo (98% hit-rate), ₹1,275/mo (100% hit-rate), ₹2,100/mo (95% hit-rate)
   - 2 active alerts (ALT-001, ALT-002)
   - 1 snoozed alert (ALT-003, snoozed until tomorrow)
   - 2 historical resolved alerts (ALT-004, ALT-005)
   - Relative-timed chronological activity feed
2. Presenter Controller fault injection studio and replay advancer.
3. Clean state reset restores pristine baseline deterministically 3 times in a row.
4. 2-Minute teleprompter 6-step catalog and script timing.
5. Pitch visuals & state-level scaling impact calculations (100, 500, 1000 facilities).
6. Main product UI cleanliness (zero demo scaffolding in app.py).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from campus_energy.decision.ledger import SavingsLedger
from campus_energy.demo.inject import clear_fault_registry, get_active_faults
from campus_energy.presenter.controller import (
    SHARED_FAULT_FILE,
    SHARED_REPLAY_FILE,
    TELEPROMPTER_STEPS,
    advance_replay_stream,
    compute_multi_campus_impact,
    get_teleprompter_step,
    reset_demo_state,
    trigger_live_fault,
)
from campus_energy.storage.repository import Repository
from campus_energy.storage.seed import (
    get_seed_activity_feed,
    get_seed_alert_statuses,
    get_seed_alerts,
    get_seed_work_orders,
    seed_database,
    seed_operational_history,
)


@pytest.fixture(autouse=True)
def clean_test_environment(tmp_path):
    """Ensure clean active fault registry and test db before and after each test."""
    clear_fault_registry()
    yield
    clear_fault_registry()


def test_seed_operational_history_populates_ledger(tmp_path):
    """Test seed_operational_history populates ledger with verified work orders, active alerts, and activity feed."""
    db_file = str(tmp_path / "test_campus.db")
    ledger = SavingsLedger()

    ref_time = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    res_ledger = seed_operational_history(
        ledger=ledger,
        facility_id="university",
        db_path=db_file,
        reference_time=ref_time,
    )

    assert res_ledger is ledger

    # 1. Verify 3 Completed & Verified Work Orders
    verified_items = ledger.get_items("verified")
    assert len(verified_items) == 3

    wo_by_id = {it.work_order_id: it for it in verified_items}
    assert "WO-2025-089" in wo_by_id
    assert "WO-2025-094" in wo_by_id
    assert "WO-2025-098" in wo_by_id

    # 1a. WO-2025-089 (LED Retrofit in Admin Block)
    wo1 = wo_by_id["WO-2025-089"]
    assert wo1.status == "verified"
    assert wo1.assignee == "Electrical Maintenance Crew"
    assert wo1.verified_savings is not None
    assert wo1.verified_savings.inr.expected == 4800.0
    assert wo1.verified_savings.kwh.expected == 565.0
    assert wo1.hit_rate_status == "hit"
    assert "98%" in str(wo1.verification_notes)

    # 1b. WO-2025-094 (Pump P1 Cavitation Seal Repair)
    wo2 = wo_by_id["WO-2025-094"]
    assert wo2.status == "verified"
    assert wo2.assignee == "Plumbing & Hydraulics Team"
    assert wo2.verified_savings is not None
    assert wo2.verified_savings.inr.expected == 1275.0
    assert wo2.verified_savings.kwh.expected == 150.0
    assert wo2.hit_rate_status == "hit"
    assert "100%" in str(wo2.verification_notes)

    # 1c. WO-2025-098 (Chiller Sensor Recalibration)
    wo3 = wo_by_id["WO-2025-098"]
    assert wo3.status == "verified"
    assert wo3.assignee == "HVAC Automation Specialist"
    assert wo3.verified_savings is not None
    assert wo3.verified_savings.inr.expected == 2100.0
    assert wo3.verified_savings.kwh.expected == 247.0
    assert wo3.hit_rate_status == "hit"
    assert "95%" in str(wo3.verification_notes)

    # 2. Verify Ledger Summary Aggregations
    summary = ledger.get_summary()
    assert summary.verified_count == 3
    assert summary.cumulative_verified_inr == 4800.0 + 1275.0 + 2100.0  # ₹8,175.0
    assert summary.cumulative_verified_kwh == 565.0 + 150.0 + 247.0    # 962.0 kWh
    assert summary.estimate_hit_rate_pct == 100.0

    # 3. Verify Active & Historical Alerts
    alerts = get_seed_alerts(ref_time)
    assert len(alerts) == 5

    active_alerts = [a for a in alerts if a["status"] == "active"]
    snoozed_alerts = [a for a in alerts if a["status"] == "snoozed"]
    resolved_alerts = [a for a in alerts if a["status"] == "resolved"]

    assert len(active_alerts) == 2
    assert {a["alert_id"] for a in active_alerts} == {"ALT-001", "ALT-002"}
    assert active_alerts[0]["anomaly_type"] == "lights_left_on"
    assert active_alerts[1]["anomaly_type"] == "pump_no_flow"

    assert len(snoozed_alerts) == 1
    assert snoozed_alerts[0]["alert_id"] == "ALT-003"
    assert snoozed_alerts[0]["snooze_until"] > ref_time

    assert len(resolved_alerts) == 2
    assert {a["alert_id"] for a in resolved_alerts} == {"ALT-004", "ALT-005"}

    # 4. Verify Alert Statuses mapping
    statuses = get_seed_alert_statuses(ref_time)
    assert statuses["ALT-001"]["status"] == "active"
    assert statuses["ALT-002"]["status"] == "active"
    assert statuses["ALT-003"]["status"] == "snoozed"
    assert statuses["ALT-004"]["status"] == "resolved"
    assert statuses["ALT-005"]["status"] == "resolved"

    # 5. Verify Chronological Activity Feed
    feed = get_seed_activity_feed(ref_time)
    assert len(feed) >= 5
    # Verify descending chronological order
    timestamps = [item["timestamp"] for item in feed]
    assert timestamps == sorted(timestamps, reverse=True)
    hints = [item.get("relative_hint") for item in feed]
    assert "15 min ago" in hints
    assert "4 hours ago" in hints
    assert "Yesterday" in hints
    assert "3 days ago" in hints

    # 6. Verify SQLite Database Repository Populated
    repo = Repository(db_path=db_file)
    db_alerts = repo.get_alerts()
    assert len(db_alerts) == 5
    db_active = repo.get_alerts(status="active")
    assert len(db_active) == 2


def test_presenter_fault_injection_and_reset(tmp_path):
    """Test presenter controller fault injection studio and clean reset action."""
    db_file = str(tmp_path / "test_campus.db")
    reset_demo_state(facility_id="university", db_path=db_file)

    assert len(get_active_faults()) == 0

    # Inject live fault: Pump No Flow / Cavitation
    mod_tel, event = trigger_live_fault(
        fault_type="pump_no_flow",
        asset_id="EQ-UTL-P1-PM",
        duration_hours=2.0,
    )

    assert event is not None
    assert event.event_type.value == "pump_no_flow"
    assert event.asset_id == "EQ-UTL-P1-PM"
    assert len(get_active_faults()) == 1

    # Verify shared fault file written
    assert SHARED_FAULT_FILE.exists()
    with open(SHARED_FAULT_FILE, "r", encoding="utf-8") as f:
        fault_json = json.load(f)
    assert fault_json["active"] is True
    assert fault_json["event_id"] == event.event_id
    assert fault_json["fault_type"] == "pump_no_flow"

    # Advance replay
    adv_res = advance_replay_stream(seconds=10, db_path=db_file)
    assert adv_res["stepped_by_seconds"] == 10
    assert adv_res["offset_seconds"] >= 10

    # Reset state
    clean_res = reset_demo_state(facility_id="university", db_path=db_file)
    assert clean_res["status"] == "success"
    assert clean_res["active_faults_count"] == 0
    assert clean_res["verified_work_orders_count"] == 3
    assert clean_res["cumulative_verified_inr"] == 8175.0
    assert len(get_active_faults()) == 0

    # Verify shared fault file marked inactive
    with open(SHARED_FAULT_FILE, "r", encoding="utf-8") as f:
        cleaned_json = json.load(f)
    assert cleaned_json["active"] is False


def test_reset_demo_deterministic_three_consecutive_runs(tmp_path):
    """Test that resetting demo restores clean seed state deterministically 3 times in a row."""
    db_file = str(tmp_path / "test_campus.db")
    ref_time = datetime(2026, 11, 10, 14, 0, 0, tzinfo=timezone.utc)

    for run_i in range(1, 4):
        # 1. Inject an arbitrary fault
        trigger_live_fault(
            fault_type="pf_drop",
            asset_id="MTR-MAIN-01",
            duration_hours=3.0,
            power_factor=0.72,
        )
        assert len(get_active_faults()) >= 1

        # 2. Advance simulated stream
        advance_replay_stream(seconds=5, db_path=db_file)

        # 3. Trigger 1-Click Clean State Reset
        res = reset_demo_state(facility_id="university", db_path=db_file, reference_time=ref_time)

        # 4. Strict assertions for deterministic baseline
        assert res["status"] == "success", f"Run {run_i} failed status"
        assert res["active_faults_count"] == 0, f"Run {run_i} left active faults"
        assert res["verified_work_orders_count"] == 3, f"Run {run_i} incorrect verified WOs"
        assert res["cumulative_verified_inr"] == 8175.0, f"Run {run_i} INR mismatch"
        assert res["cumulative_verified_kwh"] == 962.0, f"Run {run_i} kWh mismatch"
        assert res["estimate_hit_rate_pct"] == 100.0, f"Run {run_i} hit rate mismatch"
        assert res["alerts_count"] == 5, f"Run {run_i} alerts count mismatch"

        ledger = res["ledger"]
        assert len(ledger.get_items("verified")) == 3
        wo_ids = [it.work_order_id for it in ledger.get_items("verified")]
        assert wo_ids == ["WO-2025-089", "WO-2025-094", "WO-2025-098"]


def test_teleprompter_steps_catalog():
    """Test 2-Minute Presenter Script / Teleprompter covers all 6 choreographed steps."""
    assert len(TELEPROMPTER_STEPS) == 6

    step1 = get_teleprompter_step(1)
    assert step1["title"] == "Baseline Overview"
    assert step1["time_range"] == "0:00 - 0:25"
    assert "Home" in step1["screen"]
    assert "8,175" in step1["script"]

    step2 = get_teleprompter_step(2)
    assert step2["title"] == "Inject Live Failure"
    assert step2["time_range"] == "0:25 - 0:45"
    assert "Live Stream" in step2["screen"]

    step3 = get_teleprompter_step(3)
    assert step3["title"] == "Anomaly & 'Why?' Diagnosis"
    assert step3["time_range"] == "0:45 - 1:10"
    assert "Alert Drawer" in step3["screen"]

    step4 = get_teleprompter_step(4)
    assert step4["title"] == "Guarded Action & Safe Alternative"
    assert step4["time_range"] == "1:10 - 1:30"
    assert "Service-Impact Guard" in step4["screen"]

    step5 = get_teleprompter_step(5)
    assert step5["title"] == "Work Order & Live Notification"
    assert step5["time_range"] == "1:30 - 1:45"
    assert "Work Order" in step5["screen"]

    step6 = get_teleprompter_step(6)
    assert step6["title"] == "Counterfactual M&V Verification & Audit PDF"
    assert step6["time_range"] == "1:45 - 2:00"
    assert "Audit PDF" in step6["screen"]


def test_state_level_scaling_impact_scaler():
    """Test pitch visuals & state-level scaling impact calculations across institutions."""
    # Test 100 facilities (Base)
    imp100 = compute_multi_campus_impact(num_facilities=100)
    assert imp100["num_facilities"] == 100
    assert imp100["avoided_gwh"] == 4.2
    assert imp100["savings_crores_inr"] == 3.57
    assert imp100["carbon_avoided_metric_tonnes"] == 3444.0
    assert imp100["coal_avoided_tonnes"] == 2100.0

    # Test 500 facilities
    imp500 = compute_multi_campus_impact(num_facilities=500)
    assert imp500["num_facilities"] == 500
    assert imp500["avoided_gwh"] == 21.0
    assert imp500["savings_crores_inr"] == 17.85
    assert imp500["carbon_avoided_metric_tonnes"] == 17220.0
    assert imp500["coal_avoided_tonnes"] == 10500.0

    # Test 1000 facilities
    imp1000 = compute_multi_campus_impact(num_facilities=1000)
    assert imp1000["num_facilities"] == 1000
    assert imp1000["avoided_gwh"] == 42.0
    assert imp1000["savings_crores_inr"] == 35.70


def test_clean_product_ui_contains_zero_demo_scaffolding():
    """Verify that src/campus_energy/dashboard/app.py contains ZERO presentation demo scaffolding."""
    app_path = Path("src/campus_energy/dashboard/app.py")
    assert app_path.exists()
    content = app_path.read_text(encoding="utf-8")

    # 1. No "Guided Jury Tour" toggle or top banner
    assert "Guided Jury Tour" not in content
    assert "Start Guided Tour" not in content
    assert "guided_tour" not in content.lower()

    # 2. No "Jury Demo Moment" or "Jury Demo Engine"
    assert "Jury Demo Moment" not in content
    assert "Jury Demo Engine" not in content
    assert "jury" not in content.lower()

    # 3. No "Latency: 1.84 ms" brag badges
    assert "Detection Latency" not in content
    assert "Reaction latency:" not in content
    assert "1.84 ms" not in content
    assert "1.8 ms" not in content

    # 4. No state-level impact scaler in main product app
    assert "State-Level Scaling" not in content
    assert "State Facilities Adopting Nivara" not in content
