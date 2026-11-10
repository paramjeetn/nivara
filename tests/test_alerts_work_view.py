"""Unit and integration tests for Alerts & Work View (Step UX-4).

Tests:
1. Alerts filtering by severity, status, building, and asset type.
2. Alert detail data assembly (expected vs actual, impact, service-safe chip, safe alternative).
3. Inline Service-Safe Guard blocked reason and proactive safe alternative.
4. Full closed-loop operational workflow: Alert -> Create Work Order -> Complete & Verify.
5. Snooze lifecycle and automated expiration return to active.
6. False alarm marking with operator feedback and audit timeline.
7. Occupant complaints NLP clustering and quick work order dispatch.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
import pandas as pd
import pytest

from campus_energy import assumptions
from campus_energy.decision.guard import check_service_guard
from campus_energy.decision.ledger import SavingsLedger, WorkOrderItem
from campus_energy.decision.recommendation import GuardResult, Recommendation, SavingsBundle
from campus_energy.notify import NotificationRouter
from campus_energy.profiles import load_all_profiles, FacilityProfile
from campus_energy.schemas import Alert, AlertStatus, Severity
from campus_energy.ui.components import format_inr, format_relative_time, service_safe_chip
from campus_energy.ui.views.alerts_work import (
    acknowledge_alert,
    build_plain_headline,
    create_work_order_for_alert,
    filter_alerts,
    mark_false_alarm,
    prepare_alert_item,
    snooze_alert,
    verify_work_order_action,
)


@pytest.fixture
def sample_facility_profile() -> FacilityProfile:
    profiles = load_all_profiles("config/facility_profiles")
    return profiles.get("university") or list(profiles.values())[0]


@pytest.fixture
def hospital_facility_profile() -> FacilityProfile:
    profiles = load_all_profiles("config/facility_profiles")
    return profiles.get("district_hospital") or list(profiles.values())[0]


@pytest.fixture
def sample_assets_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "asset_id": "EQ-HVAC-01",
            "asset_name": "Main Chiller Unit 1",
            "asset_type": "hvac",
            "building_id": "BLD-ADMIN",
            "zone_id": "ZONE-A",
            "rated_power_kw": 25.0,
        },
        {
            "asset_id": "EQ-PUMP-02",
            "asset_name": "Rooftop Hydro Pump",
            "asset_type": "pump",
            "building_id": "BLD-UTL",
            "zone_id": "ZONE-P",
            "rated_power_kw": 7.5,
        },
        {
            "asset_id": "EQ-LT-101",
            "asset_name": "Library Highbay Lights",
            "asset_type": "lighting",
            "building_id": "BLD-LIB",
            "zone_id": "ZONE-L",
            "rated_power_kw": 4.0,
        },
        {
            "asset_id": "EQ-ICU-CHILLER",
            "asset_name": "ICU Dedicated Surgical Chiller",
            "asset_type": "hvac",
            "building_id": "BLD-HOSP-ICU",
            "zone_id": "ZONE-ICU",
            "rated_power_kw": 30.0,
        },
    ])


@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    base_time = datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc)
    rows = []
    # 2 hours of 5-min intervals
    for i in range(24):
        t = base_time + timedelta(minutes=5 * i)
        rows.append({
            "reading_id": f"RDG-{i:03d}",
            "timestamp": t.isoformat(),
            "asset_id": "EQ-HVAC-01",
            "power_kw": 22.0 if i < 12 else 4.0,
            "interval_energy_kwh": (22.0 if i < 12 else 4.0) / 12.0,
            "flow_lpm": 0.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_context_df() -> pd.DataFrame:
    base_time = datetime(2026, 11, 10, 10, 0, tzinfo=timezone.utc)
    return pd.DataFrame([
        {
            "timestamp": base_time.isoformat(),
            "building_id": "BLD-ADMIN",
            "outdoor_temperature_c": 33.5,
            "occupancy": 0,
            "is_working_hours": False,
        },
        {
            "timestamp": base_time.isoformat(),
            "building_id": "BLD-UTL",
            "outdoor_temperature_c": 33.5,
            "occupancy": 0,
            "is_working_hours": False,
        },
    ])


@pytest.fixture
def sample_alerts() -> List[Alert]:
    now = datetime.now(timezone.utc)
    return [
        Alert(
            alert_id="ALT-001",
            detected_at=now - timedelta(minutes=15),
            asset_id="EQ-HVAC-01",
            building_id="BLD-ADMIN",
            anomaly_type="hvac_left_on",
            severity=Severity.CRITICAL,
            actual_value=24.0,
            expected_value=4.0,
            deviation=20.0,
            evidence="Observed 24.0 kW draw during unoccupied night hours",
            recommendation="Dispatch technician to check setback timer schedule",
            confidence=0.94,
        ),
        Alert(
            alert_id="ALT-002",
            detected_at=now - timedelta(hours=1),
            asset_id="EQ-PUMP-02",
            building_id="BLD-UTL",
            anomaly_type="pump_no_flow",
            severity=Severity.HIGH,
            actual_value=7.5,
            expected_value=0.0,
            deviation=7.5,
            evidence="Pump operating at rated power with 0 LPM measured water flow",
            recommendation="Inspect pump suction valve and pressure transducer",
            confidence=0.91,
        ),
        Alert(
            alert_id="ALT-003",
            detected_at=now - timedelta(hours=3),
            asset_id="EQ-LT-101",
            building_id="BLD-LIB",
            anomaly_type="lights_left_on",
            severity=Severity.MEDIUM,
            actual_value=4.0,
            expected_value=0.0,
            deviation=4.0,
            evidence="Highbay lights burning in closed library",
            recommendation="Turn off breaker or reset occupancy PIR sensor",
            confidence=0.88,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Headline & Data Assembly Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_build_plain_headline():
    """Verify single plain-language operational headline generation with expected vs actual numbers."""
    h_pump = build_plain_headline(
        asset_id="EQ-PUMP-02",
        asset_name="Rooftop Hydro Pump",
        anomaly_type="pump_no_flow",
        actual_val=7.5,
        expected_val=0.0,
    )
    assert "Rooftop Hydro Pump" in h_pump
    assert "7.5 kW" in h_pump
    assert "0 LPM" in h_pump
    assert "expected 0.0 kW" in h_pump

    h_hvac = build_plain_headline(
        asset_id="EQ-HVAC-01",
        asset_name="Main Chiller Unit 1",
        anomaly_type="hvac_left_on",
        actual_val=25.0,
        expected_val=3.0,
    )
    assert "25.0 kW" in h_hvac
    assert "unoccupied hours" in h_hvac
    assert "3.0 kW" in h_hvac


def test_prepare_alert_item(sample_alerts, sample_assets_df, sample_context_df, sample_facility_profile):
    """Verify alert item data contract preparation (values, impact, explainability, service guard)."""
    item = prepare_alert_item(
        alert=sample_alerts[0],
        assets_df=sample_assets_df,
        context_df=sample_context_df,
        facility_meta=sample_facility_profile,
    )

    assert item["alert_id"] == "ALT-001"
    assert item["asset_name"] == "Main Chiller Unit 1"
    assert item["actual_value"] == 24.0
    assert item["expected_value"] == 4.0
    assert item["deviation"] == 20.0
    assert item["est_waste_kwh_day"] > 0
    assert item["est_waste_inr_day"] > 0
    assert "Main Chiller Unit 1" in item["headline"]

    # Explainability
    expl = item["explainability"]
    assert "detector_type" in expl
    assert "why_headline" in expl
    assert "probable_root_cause" in expl
    assert "recommended_verification" in expl

    # Recommendation & Service-Impact Guard
    rec = item["recommendation"]
    assert rec is not None
    assert rec.service_guard is not None
    assert rec.service_guard.status in ("pass", "needs_review", "blocked")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Filtering Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_alerts_by_severity(sample_alerts, sample_assets_df, sample_facility_profile):
    """Test filtering alert inbox items by severity levels."""
    items = [prepare_alert_item(a, sample_assets_df, facility_meta=sample_facility_profile) for a in sample_alerts]

    # Filter critical
    crit_filtered = filter_alerts(items, severity_filter="Critical")
    assert len(crit_filtered) == 1
    assert crit_filtered[0]["alert_id"] == "ALT-001"

    # Filter high
    high_filtered = filter_alerts(items, severity_filter="High")
    assert len(high_filtered) == 1
    assert high_filtered[0]["alert_id"] == "ALT-002"

    # Filter All
    all_filtered = filter_alerts(items, severity_filter="All")
    assert len(all_filtered) == 3


def test_filter_alerts_by_building_and_asset_type(sample_alerts, sample_assets_df, sample_facility_profile):
    """Test filtering alert inbox items by building and asset type."""
    items = [prepare_alert_item(a, sample_assets_df, facility_meta=sample_facility_profile) for a in sample_alerts]

    # Filter Building
    bld_filtered = filter_alerts(items, building_filter="BLD-ADMIN")
    assert len(bld_filtered) == 1
    assert bld_filtered[0]["building_id"] == "BLD-ADMIN"

    # Filter Asset Type
    pump_filtered = filter_alerts(items, asset_type_filter="Pump")
    assert len(pump_filtered) == 1
    assert pump_filtered[0]["asset_type"] == "pump"


def test_filter_alerts_by_lifecycle_status(sample_alerts, sample_assets_df, sample_facility_profile):
    """Test filtering alert inbox items by lifecycle status (active, snoozed, closed)."""
    items = [prepare_alert_item(a, sample_assets_df, facility_meta=sample_facility_profile) for a in sample_alerts]
    statuses = {
        "ALT-001": {"status": "active"},
        "ALT-002": {"status": "snoozed", "snooze_until": datetime.now(timezone.utc) + timedelta(hours=2)},
        "ALT-003": {"status": "closed", "feedback": "Maintenance testing"},
    }

    active_filtered = filter_alerts(items, status_filter="Active", alert_statuses=statuses)
    assert len(active_filtered) == 1
    assert active_filtered[0]["alert_id"] == "ALT-001"

    snoozed_filtered = filter_alerts(items, status_filter="Snoozed", alert_statuses=statuses)
    assert len(snoozed_filtered) == 1
    assert snoozed_filtered[0]["alert_id"] == "ALT-002"

    closed_filtered = filter_alerts(items, status_filter="Closed", alert_statuses=statuses)
    assert len(closed_filtered) == 1
    assert closed_filtered[0]["alert_id"] == "ALT-003"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Service-Safe Inline Integration & Safe Alternative Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_service_guard_blocked_inline_alternative(hospital_facility_profile, sample_assets_df):
    """Test inline Service-Safe Guard blocked status with clinical rationale and safe alternative."""
    # Create an alert on an ICU load in a hospital setting
    icu_alert = Alert(
        alert_id="ALT-ICU-999",
        detected_at=datetime.now(timezone.utc),
        asset_id="EQ-IPD-ICU-HVAC",
        building_id="BLD-HOSP-IPD",
        anomaly_type="peak_shaving",
        severity=Severity.CRITICAL,
        actual_value=32.0,
        expected_value=10.0,
        deviation=22.0,
        evidence="High demand during peak tariff window",
        recommendation="Curtail ICU chiller power by 40% during peak tariff window",
        confidence=0.95,
    )

    item = prepare_alert_item(
        alert=icu_alert,
        assets_df=sample_assets_df,
        facility_meta=hospital_facility_profile,
    )

    guard = item["service_guard"]
    assert guard.status == "blocked"
    assert "protected" in guard.reason.lower() or "icu" in guard.reason.lower() or "clinical" in guard.reason.lower()
    assert guard.safe_alternative is not None
    assert len(guard.safe_alternative) > 10

    # Test that service_safe_chip generates the red blocked badge
    chip_html = service_safe_chip(guard.status, guard.reason, guard.safe_alternative)
    assert "chip-service-safe-blocked" in chip_html
    assert "Not recommended" in chip_html
    assert "Reason:" in chip_html
    assert "Safe Alternative:" in chip_html


# ─────────────────────────────────────────────────────────────────────────────
# 4. Full Operational Closed-Loop Workflow Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_full_operational_workflow_alert_to_wo_to_verify(
    sample_alerts,
    sample_assets_df,
    sample_context_df,
    sample_facility_profile,
    sample_telemetry_df,
):
    """Test end-to-end operational path: Alert -> Create Work Order -> Complete & Verify."""
    ledger = SavingsLedger()
    router = NotificationRouter(dry_run=True)
    statuses: Dict[str, Any] = {}

    item = prepare_alert_item(
        alert=sample_alerts[0],
        assets_df=sample_assets_df,
        context_df=sample_context_df,
        facility_meta=sample_facility_profile,
    )

    # 1. Dispatch Work Order
    wo_item = create_work_order_for_alert(
        alert_item=item,
        ledger=ledger,
        assignee="HVAC Lead",
        sla_hours=12,
        notif_router=router,
        alert_statuses=statuses,
    )

    assert wo_item is not None
    assert wo_item.work_order_id.startswith("WO-")
    assert wo_item.status == "work_order"
    assert wo_item.assignee == "HVAC Lead"
    assert wo_item.sla_deadline is not None

    # Check alert tracking
    aid = item["alert_id"]
    assert statuses[aid]["work_order_id"] == wo_item.work_order_id
    timeline = statuses[aid]["timeline"]
    assert len(timeline) >= 1
    assert any("created and assigned to HVAC Lead" in ev["event"] for ev in timeline)

    # 2. Complete & Verify via M&V Counterfactual Baseline
    verified_wo = verify_work_order_action(
        wo_id=wo_item.work_order_id,
        ledger=ledger,
        telemetry_df=sample_telemetry_df,
        alert_statuses=statuses,
    )

    assert verified_wo.status == "verified"
    assert verified_wo.is_verified is True
    assert verified_wo.verified_savings is not None
    assert verified_wo.verified_savings.inr.expected >= 0.0
    assert verified_wo.hit_rate_status in ("hit", "exceeded", "underperformed")

    # Check alert was updated to resolved
    assert statuses[aid]["status"] == "resolved"
    assert any("verified via IPMVP Option B" in ev["event"] for ev in statuses[aid]["timeline"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Snooze and False Alarm Lifecycle Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_snooze_alert_and_expiration():
    """Test snoozing an alert and verifying expiration handling."""
    statuses: Dict[str, Any] = {}
    aid = "ALT-SNOOZE-TEST"

    # Snooze for 2 hours
    entry = snooze_alert(aid, duration_hours=2, alert_statuses=statuses)
    assert entry["status"] == "snoozed"
    assert entry["snooze_until"] > datetime.now(timezone.utc)
    assert any("snoozed for 2h" in ev["event"] for ev in entry["timeline"])

    # Simulate expired snooze
    entry["snooze_until"] = datetime.now(timezone.utc) - timedelta(minutes=5)

    items = [{"alert_id": aid, "status": "active", "severity": "high", "building_id": "BLD-1", "asset_type": "hvac"}]
    filtered = filter_alerts(items, status_filter="Active", alert_statuses=statuses)
    assert len(filtered) == 1
    assert statuses[aid]["status"] == "active"


def test_mark_false_alarm_feedback():
    """Test recording false alarm operator feedback and closing alert."""
    statuses: Dict[str, Any] = {}
    aid = "ALT-FA-TEST"

    entry = mark_false_alarm(
        alert_id=aid,
        feedback_reason="Known routine chiller maintenance testing",
        alert_statuses=statuses,
        operator_name="Senior Engineer",
    )

    assert entry["status"] == "closed"
    assert entry["feedback"] == "Known routine chiller maintenance testing"
    timeline = entry["timeline"]
    assert any("Closed as False Alarm by Senior Engineer" in ev["event"] for ev in timeline)


def test_acknowledge_alert():
    """Test acknowledging an alert."""
    statuses: Dict[str, Any] = {}
    aid = "ALT-ACK-TEST"

    entry = acknowledge_alert(aid, alert_statuses=statuses, operator_name="Shift Operator")
    assert entry["status"] == "acknowledged"
    assert any("Acknowledged by Shift Operator" in ev["event"] for ev in entry["timeline"])
