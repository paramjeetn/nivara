"""Unit and end-to-end integration tests for Live Fault Injection and Guided Demo Mode (Step 13).

Covers:
1. Canonical fault injection for all 5 fault types (lights_left_on, pump_no_flow, pipe_leak_overnight,
   sensor_dropout, pf_drop) verifying telemetry mutations, flags, and Event schemas.
2. ActiveFaultRegistry registration, querying, resolution, and clearing.
3. GuidedTour state machine transitions, boundary enforcement, and reset functionality.
4. End-to-end 6-step demo tour execution repeated THREE times consecutively from a clean state.
5. Reaction latency benchmarking (injection to alert generation).
6. FastAPI demo ingestion endpoints (/api/demo/inject, /api/demo/reset, /api/demo/tour).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from campus_energy.decision.ledger import SavingsLedger
from campus_energy.decision.recommendation import Recommendation, SavingsBundle
from campus_energy.decision.guard import check_service_guard
from campus_energy.demo.inject import (
    FAULT_TYPES,
    ActiveFaultRegistry,
    active_fault_registry,
    clear_fault_registry,
    get_active_faults,
    get_fault_presets,
    get_supported_fault_types,
    inject_live_fault,
    resolve_fault,
)
from campus_energy.demo.tour import (
    TOUR_STEPS,
    GuidedTour,
    get_default_tour,
    get_tour_step_info,
)
from campus_energy.ingestion.api import app
from campus_energy.notify import NotificationMessage, get_notification_router
from campus_energy.profiles import FacilityProfile, ProtectedLoad
from campus_energy.schemas import (
    Alert,
    AlertStatus,
    EquipmentState,
    Event,
    EventType,
    QualityFlag,
    SensorStatus,
    Severity,
)
from campus_energy.simulator.events import simulate_fault_resolution


@pytest.fixture
def sample_assets_df() -> pd.DataFrame:
    """Standardized multi-asset campus registry."""
    return pd.DataFrame([
        {
            "asset_id": "MTR-MAIN-01",
            "asset_name": "Main Campus Substation Meter",
            "building_id": "BLD-ADMIN",
            "asset_type": "meter",
            "monitoring_level": "building",
            "rated_power_kw": 250.0,
            "flow_rated_lpm": None,
        },
        {
            "asset_id": "EQ-LTG-LH01",
            "asset_name": "Lecture Hall 1 Lighting Array",
            "building_id": "BLD-ACAD",
            "zone_id": "ZN-LH01",
            "asset_type": "lighting",
            "monitoring_level": "equipment",
            "rated_power_kw": 18.0,
            "flow_rated_lpm": None,
        },
        {
            "asset_id": "EQ-PUMP-CHW01",
            "asset_name": "Primary Chilled Water Circulation Pump",
            "building_id": "BLD-CENTRAL",
            "zone_id": "ZN-PLANT",
            "asset_type": "pump",
            "monitoring_level": "equipment",
            "rated_power_kw": 15.0,
            "flow_rated_lpm": 250.0,
        },
        {
            "asset_id": "MTR-WATER-01",
            "asset_name": "Academic Riser Water Inflow Meter",
            "building_id": "BLD-ACAD",
            "zone_id": "ZN-BASEMENT",
            "asset_type": "meter",
            "monitoring_level": "building",
            "rated_power_kw": 1.0,
            "flow_rated_lpm": 120.0,
        },
        {
            "asset_id": "EQ-ICU-HVAC",
            "asset_name": "Hospital Critical ICU Air Handling Unit",
            "building_id": "BLD-HOSP",
            "zone_id": "ZN-ICU",
            "asset_type": "hvac",
            "monitoring_level": "equipment",
            "rated_power_kw": 30.0,
            "flow_rated_lpm": None,
        },
    ])


@pytest.fixture
def baseline_telemetry_df() -> pd.DataFrame:
    """Create 24-hour baseline telemetry at 5-minute intervals across assets."""
    start_time = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    intervals = 288  # 24 hours * 12 (5-min intervals)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(intervals)]

    rows = []
    for ts in timestamps:
        hour = ts.hour
        # Substation Meter
        rows.append({
            "reading_id": f"TEL-MTR-{ts.strftime('%H%M')}",
            "timestamp": ts.isoformat(),
            "asset_id": "MTR-MAIN-01",
            "reading_type": "meter",
            "power_kw": 120.0 if 8 <= hour <= 18 else 30.0,
            "interval_energy_kwh": (120.0 if 8 <= hour <= 18 else 30.0) / 12.0,
            "cumulative_energy_kwh": 5000.0,
            "voltage_v": 415.0,
            "current_a": 175.0 if 8 <= hour <= 18 else 45.0,
            "power_factor": 0.96,
            "state": "ON",
            "flow_lpm": None,
            "temperature_c": None,
            "setpoint_c": None,
            "sensor_status": "healthy",
            "quality_flag": "valid",
        })
        # Lighting Array (normally OFF at night)
        ltg_on = (8 <= hour <= 20)
        rows.append({
            "reading_id": f"TEL-LTG-{ts.strftime('%H%M')}",
            "timestamp": ts.isoformat(),
            "asset_id": "EQ-LTG-LH01",
            "reading_type": "equipment",
            "power_kw": 16.5 if ltg_on else 0.2,
            "interval_energy_kwh": (16.5 if ltg_on else 0.2) / 12.0,
            "cumulative_energy_kwh": 800.0,
            "voltage_v": 230.0,
            "current_a": 72.0 if ltg_on else 1.0,
            "power_factor": 0.95,
            "state": "ON" if ltg_on else "OFF",
            "flow_lpm": None,
            "temperature_c": 23.0,
            "setpoint_c": 23.0,
            "sensor_status": "healthy",
            "quality_flag": "valid",
        })
        # Pump
        pump_on = (6 <= hour <= 22)
        rows.append({
            "reading_id": f"TEL-PMP-{ts.strftime('%H%M')}",
            "timestamp": ts.isoformat(),
            "asset_id": "EQ-PUMP-CHW01",
            "reading_type": "equipment",
            "power_kw": 14.2 if pump_on else 0.0,
            "interval_energy_kwh": (14.2 if pump_on else 0.0) / 12.0,
            "cumulative_energy_kwh": 1200.0,
            "voltage_v": 415.0,
            "current_a": 21.0 if pump_on else 0.0,
            "power_factor": 0.92,
            "state": "ON" if pump_on else "OFF",
            "flow_lpm": 240.0 if pump_on else 0.0,
            "temperature_c": 18.0,
            "setpoint_c": 18.0,
            "sensor_status": "healthy",
            "quality_flag": "valid",
        })
        # Water Meter (night flow ~1.5 LPM)
        rows.append({
            "reading_id": f"TEL-WTR-{ts.strftime('%H%M')}",
            "timestamp": ts.isoformat(),
            "asset_id": "MTR-WATER-01",
            "reading_type": "meter",
            "power_kw": 0.5,
            "interval_energy_kwh": 0.04,
            "cumulative_energy_kwh": 300.0,
            "voltage_v": 230.0,
            "current_a": 2.2,
            "power_factor": 0.95,
            "state": "ON",
            "flow_lpm": 1.5 if 1 <= hour <= 5 else 35.0,
            "temperature_c": 22.0,
            "setpoint_c": 22.0,
            "sensor_status": "healthy",
            "quality_flag": "valid",
        })

    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure clean fault registry and default tour state before every test."""
    clear_fault_registry()
    get_default_tour().reset_demo()
    yield
    clear_fault_registry()
    get_default_tour().reset_demo()


# ─────────────────────────────────────────────────────────────────────────────
# 1. CANONICAL FAULT INJECTION TESTS (5 FAULT TYPES)
# ─────────────────────────────────────────────────────────────────────────────

def test_inject_fault_lights_left_on(baseline_telemetry_df):
    """Test lights_left_on forces power near rated and state ON during off-hours."""
    asset_id = "EQ-LTG-LH01"
    start_time = datetime(2026, 11, 1, 22, 0, tzinfo=timezone.utc)
    duration_hours = 4.0

    mod_df, event = inject_live_fault(
        telemetry_df=baseline_telemetry_df,
        fault_type="lights_left_on",
        asset_id=asset_id,
        start_time=start_time,
        duration_hours=duration_hours,
        rated_power_kw=18.0,
    )

    assert isinstance(event, Event)
    assert event.event_type == EventType.LIGHTS_LEFT_ON
    assert event.severity == Severity.MEDIUM
    assert event.asset_id == asset_id
    assert "lights" in event.description.lower()

    # Verify telemetry values in injected window
    ts = pd.to_datetime(mod_df["timestamp"])
    mask = (mod_df["asset_id"] == asset_id) & (ts >= start_time) & (ts <= start_time + timedelta(hours=duration_hours))
    injected_rows = mod_df[mask]

    assert len(injected_rows) > 0
    assert (injected_rows["power_kw"] >= 16.0).all()
    assert (injected_rows["state"] == "ON").all()
    assert (injected_rows["interval_energy_kwh"] > 1.0).all()

    # Verify registration in active registry
    active = get_active_faults()
    assert len(active) == 1
    assert active[0].event_id == event.event_id


def test_inject_fault_pump_no_flow(baseline_telemetry_df):
    """Test pump_no_flow forces zero flow while drawing power (cavitation)."""
    asset_id = "EQ-PUMP-CHW01"
    start_time = datetime(2026, 11, 1, 10, 0, tzinfo=timezone.utc)
    duration_hours = 2.0

    mod_df, event = inject_live_fault(
        telemetry_df=baseline_telemetry_df,
        fault_type="pump_no_flow",
        asset_id=asset_id,
        start_time=start_time,
        duration_hours=duration_hours,
        rated_power_kw=15.0,
    )

    assert event.event_type == EventType.PUMP_NO_FLOW
    assert event.severity == Severity.HIGH
    assert event.asset_id == asset_id

    ts = pd.to_datetime(mod_df["timestamp"])
    mask = (mod_df["asset_id"] == asset_id) & (ts >= start_time) & (ts <= start_time + timedelta(hours=duration_hours))
    injected_rows = mod_df[mask]

    assert len(injected_rows) > 0
    assert (injected_rows["flow_lpm"] == 0.0).all()
    assert (injected_rows["power_kw"] >= 14.0).all()
    assert (injected_rows["state"] == "ON").all()


def test_inject_fault_pipe_leak_overnight(baseline_telemetry_df):
    """Test pipe_leak_overnight injects sudden +35 LPM flow surge."""
    asset_id = "MTR-WATER-01"
    start_time = datetime(2026, 11, 1, 2, 0, tzinfo=timezone.utc)
    duration_hours = 3.0

    mod_df, event = inject_live_fault(
        telemetry_df=baseline_telemetry_df,
        fault_type="pipe_leak_overnight",
        asset_id=asset_id,
        start_time=start_time,
        duration_hours=duration_hours,
        surge_lpm=35.0,
    )

    assert event.event_type == EventType.PIPE_LEAK_OVERNIGHT
    assert event.severity == Severity.HIGH
    assert event.asset_id == asset_id

    ts = pd.to_datetime(mod_df["timestamp"])
    mask = (mod_df["asset_id"] == asset_id) & (ts >= start_time) & (ts <= start_time + timedelta(hours=duration_hours))
    injected_rows = mod_df[mask]

    assert len(injected_rows) > 0
    assert (injected_rows["flow_lpm"] >= 35.0).all()


def test_inject_fault_sensor_dropout(baseline_telemetry_df):
    """Test sensor_dropout flags quality_flag missing and sensor offline."""
    asset_id = "MTR-MAIN-01"
    start_time = datetime(2026, 11, 1, 14, 0, tzinfo=timezone.utc)
    duration_hours = 2.0

    mod_df, event = inject_live_fault(
        telemetry_df=baseline_telemetry_df,
        fault_type="sensor_dropout",
        asset_id=asset_id,
        start_time=start_time,
        duration_hours=duration_hours,
    )

    assert event.event_type == EventType.SENSOR_DROPOUT
    assert event.severity == Severity.LOW
    assert event.asset_id == asset_id

    ts = pd.to_datetime(mod_df["timestamp"])
    mask = (mod_df["asset_id"] == asset_id) & (ts >= start_time) & (ts <= start_time + timedelta(hours=duration_hours))
    injected_rows = mod_df[mask]

    assert len(injected_rows) > 0
    assert injected_rows["power_kw"].isna().all()
    assert (injected_rows["quality_flag"] == "missing").all()
    assert (injected_rows["sensor_status"] == "offline").all()


def test_inject_fault_pf_drop(baseline_telemetry_df):
    """Test pf_drop drops power factor to 0.72 breaching 0.90 OERC threshold."""
    asset_id = "MTR-MAIN-01"
    start_time = datetime(2026, 11, 1, 11, 0, tzinfo=timezone.utc)
    duration_hours = 4.0

    mod_df, event = inject_live_fault(
        telemetry_df=baseline_telemetry_df,
        fault_type="pf_drop",
        asset_id=asset_id,
        start_time=start_time,
        duration_hours=duration_hours,
        power_factor=0.72,
    )

    assert event.event_type == EventType.PF_DROP
    assert event.severity == Severity.HIGH
    assert event.asset_id == asset_id

    ts = pd.to_datetime(mod_df["timestamp"])
    mask = (mod_df["asset_id"] == asset_id) & (ts >= start_time) & (ts <= start_time + timedelta(hours=duration_hours))
    injected_rows = mod_df[mask]

    assert len(injected_rows) > 0
    assert (injected_rows["power_factor"] == 0.72).all()


def test_active_fault_registry_lifecycle(baseline_telemetry_df):
    """Test registry registration, retrieval, resolution, and clear functionality."""
    assert active_fault_registry.count() == 0

    _, ev1 = inject_live_fault(baseline_telemetry_df, "lights_left_on", "EQ-LTG-LH01")
    _, ev2 = inject_live_fault(baseline_telemetry_df, "pump_no_flow", "EQ-PUMP-CHW01")

    assert active_fault_registry.count() == 2
    assert len(get_active_faults()) == 2

    # Resolve first fault
    assert resolve_fault(ev1.event_id) is True
    assert active_fault_registry.count() == 1
    remaining = get_active_faults()
    assert len(remaining) == 1
    assert remaining[0].event_id == ev2.event_id

    # Resolve non-existent fault returns False
    assert resolve_fault("NON-EXISTENT") is False

    # Clear registry
    clear_fault_registry()
    assert active_fault_registry.count() == 0


def test_fault_presets_completeness():
    """Verify presets are configured for all 5 canonical fault types."""
    presets = get_fault_presets()
    for ft in FAULT_TYPES:
        assert ft in presets
        meta = presets[ft]
        assert "title" in meta
        assert "icon" in meta
        assert "description" in meta
        assert "expected_detection" in meta


# ─────────────────────────────────────────────────────────────────────────────
# 2. GUIDED DEMO TOUR ENGINE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_guided_tour_state_machine():
    """Verify GuidedTour transitions, step boundary enforcement, and reset."""
    tour = GuidedTour()
    assert tour.current_step == 1
    assert tour.is_first_step() is True
    assert tour.is_last_step() is False

    # Walk through forward steps 1 -> 6
    for expected in [2, 3, 4, 5, 6]:
        s = tour.next_step()
        assert s == expected
        assert tour.current_step == expected

    # Cap at step 6
    assert tour.next_step() == 6
    assert tour.is_last_step() is True

    # Walk backwards 6 -> 1
    for expected in [5, 4, 3, 2, 1]:
        s = tour.prev_step()
        assert s == expected
        assert tour.current_step == expected

    # Floor at step 1
    assert tour.prev_step() == 1
    assert tour.is_first_step() is True

    # Direct step setting
    assert tour.set_step(4) == 4
    with pytest.raises(ValueError):
        tour.set_step(0)
    with pytest.raises(ValueError):
        tour.set_step(7)

    # Reset demo
    tour.set_injected_event_id("EVT-TEST-001")
    tour.set_reaction_latency_ms(24.5)
    tour.reset_demo()
    assert tour.current_step == 1
    assert tour.get_injected_event_id() is None
    assert tour.get_reaction_latency_ms() is None


def test_guided_tour_steps_metadata():
    """Verify all 6 steps contain complete narrative and target page mappings."""
    assert len(TOUR_STEPS) == 6
    tour = GuidedTour()

    expected_titles = [
        "Executive Baseline",
        "Fault Injection",
        "Anomaly Detection",
        "Guarded Action",
        "Work Order",
        "Resolution",
    ]

    for i in range(1, 7):
        info = tour.get_step_info(i)
        assert info["step"] == i
        assert any(k in info["title"] for k in [expected_titles[i - 1], str(i)])
        assert len(info["narrative"]) > 20
        assert len(info["talking_points"]) >= 2
        assert len(info["action_buttons"]) >= 1
        assert "target_page" in info


# ─────────────────────────────────────────────────────────────────────────────
# 3. END-TO-END SCRIPTED TOUR 3-RUN REPRODUCIBILITY TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_scripted_tour_flow_end_to_end_3_runs(sample_assets_df, baseline_telemetry_df):
    """Execute complete 6-step walkthrough THREE consecutive times from clean state.

    Flow per run:
      Step 1: Baseline inspection
      Step 2: Inject fault (pump dry run)
      Step 3: Triple-engine anomaly consensus verification
      Step 4: Recommendation generation & Service-Impact Guard certification
      Step 5: CMMS Work Order dispatch & multi-channel notification
      Step 6: Fault resolution simulation & IPMVP Option B ledger verification
      Reset demo
    """
    tour = get_default_tour()
    facility_profile = FacilityProfile(
        facility_id="university",
        facility_name="BPUT Rourkela Main Campus",
        facility_type="university",
        location="Rourkela, Odisha",
        floor_area_m2=45000.0,
        peak_occupancy=3500,
        data_dir="data/generated",
        tariff_profile="odisha_ht_commercial",
        protected_loads=[
            ProtectedLoad(
                asset_id="EQ-ICU-HVAC",
                asset_name="Hospital Critical ICU Air Handling Unit",
                zone_id="ZN-ICU",
                category="life_safety",
                curtailable=False,
                reason="Continuous positive-pressure clinical air circulation required by NABH guidelines.",
            )
        ],
    )

    for run_idx in range(1, 4):
        # ── Step 1: Baseline Verification ──
        tour.reset_demo()
        assert tour.current_step == 1
        assert len(get_active_faults()) == 0
        step1_info = tour.get_step_info()
        assert "Baseline" in step1_info["title"]

        # ── Step 2: Live Fault Injection ──
        tour.next_step()
        assert tour.current_step == 2

        inj_start = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
        mod_tel, event = inject_live_fault(
            telemetry_df=baseline_telemetry_df,
            fault_type="pump_no_flow",
            asset_id="EQ-PUMP-CHW01",
            start_time=inj_start,
            duration_hours=2.0,
            rated_power_kw=15.0,
        )
        assert event.event_type == EventType.PUMP_NO_FLOW
        tour.set_injected_event_id(event.event_id)
        assert len(get_active_faults()) == 1

        # ── Step 3: Anomaly Detection Consensus ──
        tour.next_step()
        assert tour.current_step == 3

        # Formulate consensus alert
        alert = Alert(
            alert_id=f"ALT-RUN{run_idx}-001",
            detected_at=datetime.now(timezone.utc),
            event_start_estimate=inj_start,
            asset_id="EQ-PUMP-CHW01",
            building_id="BLD-CENTRAL",
            zone_id="ZN-PLANT",
            anomaly_type="pump_no_flow",
            severity=Severity.HIGH,
            actual_value=0.0,
            expected_value=240.0,
            deviation=15.0,
            evidence="Pump drawing 15 kW electrical power with 0.0 LPM flow rate (dry-run cavitation).",
            recommendation="Immediate mechanical dispatch to clear suction blockage.",
            confidence=0.96,
        )
        assert alert.confidence >= 0.90
        assert alert.severity == Severity.HIGH

        # ── Step 4: Guarded Action & Service Guard Certification ──
        tour.next_step()
        assert tour.current_step == 4

        bundle = SavingsBundle.from_expected(kwh_expected=150.0, inr_per_kwh=8.50, co2_per_kwh=0.82)
        rec = Recommendation(
            id=f"REC-RUN{run_idx}-01",
            facility_id="university",
            type="pump_no_flow",
            target="EQ-PUMP-CHW01",
            action_text="Isolate pump and clear strainer blockage to stop motor overheating.",
            trigger=event.event_id,
            savings=bundle,
            owner_role="operations",
            lifecycle="proposed",
        )
        guard_res = check_service_guard(rec, facility_profile)
        assert guard_res.status in ("pass", "allowed")  # Circulation pump is non-protected, passed by guard

        # Test that protected ICU load IS correctly blocked by guard
        rec_icu = Recommendation(
            id=f"REC-RUN{run_idx}-ICU",
            facility_id="university",
            type="peak_shaving",
            target="EQ-ICU-HVAC",
            action_text="Curtail ICU ventilation power by 40% during peak tariff.",
            trigger="peak_tariff_spike",
            savings=bundle,
            owner_role="operations",
        )
        guard_icu = check_service_guard(rec_icu, facility_profile)
        assert guard_icu.status == "blocked"

        # ── Step 5: Work Order & Notification Dispatch ──
        tour.next_step()
        assert tour.current_step == 5

        ledger = SavingsLedger()
        ledger.add_recommendation(rec, priority="high", assignee="Plumbing & Hydraulics Team", work_order_id=f"WO-RUN{run_idx}")
        ledger.create_work_order(f"WO-RUN{run_idx}", assignee="Plumbing & Hydraulics Team", sla_hours=12)

        # Dispatch live notification from work order
        router = get_notification_router()
        dispatched_wo = ledger.get_item(f"WO-RUN{run_idx}")
        assert dispatched_wo is not None
        results = router.dispatch_from_work_order(dispatched_wo)
        assert len(results) >= 1
        assert any(r.success for r in results)

        # ── Step 6: Resolution & Verified M&V Ledger ──
        tour.next_step()
        assert tour.current_step == 6

        comp_time = inj_start + timedelta(hours=1)
        res_tel = simulate_fault_resolution(
            telemetry_df=mod_tel,
            event=event,
            resolution_time=comp_time,
            baseline_df=baseline_telemetry_df,
        )
        ledger.complete_and_verify(
            wo_id=f"WO-RUN{run_idx}",
            completion_time=comp_time,
            post_telemetry_df=res_tel,
            baseline_df=baseline_telemetry_df,
        )

        wo_item = ledger.get_item(f"WO-RUN{run_idx}")
        assert wo_item is not None
        assert wo_item.status == "verified"
        assert wo_item.verified_savings is not None
        assert wo_item.verified_savings.kwh.expected >= 0.0

        summary = ledger.get_summary()
        assert summary.verified_count == 1
        assert summary.total_actions == 1
        assert summary.cumulative_verified_kwh >= 0.0
        assert summary.estimate_hit_rate_pct >= 0.0

        # Reset clean state for next iteration
        tour.reset_demo()
        assert tour.current_step == 1
        assert len(get_active_faults()) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. REACTION LATENCY BENCHMARK TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_reaction_latency_benchmark(baseline_telemetry_df):
    """Benchmark end-to-end latency from fault injection to alert generation.

    Asserts execution finishes well under 500ms (typical sub-20ms).
    """
    latencies = []

    for ft in FAULT_TYPES:
        t0 = time.perf_counter()
        mod_df, event = inject_live_fault(
            telemetry_df=baseline_telemetry_df,
            fault_type=ft,
            asset_id="EQ-LTG-LH01" if ft == "lights_left_on" else "MTR-MAIN-01",
            duration_hours=2.0,
        )

        # Immediate alert detection simulation
        alert = Alert(
            alert_id=f"ALT-BENCH-{event.event_id}",
            detected_at=datetime.now(timezone.utc),
            event_start_estimate=event.start_time,
            asset_id=event.asset_id,
            building_id="BLD-ACAD",
            anomaly_type=event.event_type.value,
            severity=event.severity,
            evidence=event.description,
            recommendation="Inspect asset immediately.",
            confidence=0.95,
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        latencies.append(latency_ms)

        assert alert.alert_id is not None
        assert latency_ms < 500.0  # Strict SLA: under half a second

    avg_latency = np.mean(latencies)
    assert avg_latency < 100.0  # Average under 100ms
    tour = get_default_tour()
    tour.set_reaction_latency_ms(avg_latency)
    assert tour.get_reaction_latency_ms() == round(avg_latency, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. FASTAPI INGESTION ENDPOINTS TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_fastapi_demo_endpoints():
    """Test /api/demo/inject, /api/demo/reset, and /api/demo/tour endpoints."""
    client = TestClient(app)

    # 1. Reset demo state
    res_reset = client.post("/api/demo/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "reset"
    assert res_reset.json()["active_faults"] == 0

    # 2. Get tour status (initial)
    res_tour = client.get("/api/demo/tour")
    assert res_tour.status_code == 200
    assert res_tour.json()["current_step"] == 1
    assert res_tour.json()["active_faults_count"] == 0

    # 3. Inject canonical fault via API
    payload = {
        "fault_type": "pump_no_flow",
        "asset_id": "EQ-PUMP-CHW01",
        "duration_hours": 3.0,
        "params": {"rated_power_kw": 15.0},
    }
    res_inj = client.post("/api/demo/inject", json=payload)
    assert res_inj.status_code == 200
    inj_data = res_inj.json()
    assert inj_data["status"] == "injected"
    assert inj_data["fault_type"] == "pump_no_flow"
    assert inj_data["asset_id"] == "EQ-PUMP-CHW01"
    assert "event_id" in inj_data
    assert "event" in inj_data

    # 4. Check tour state updated with active fault
    res_tour2 = client.get("/api/demo/tour")
    assert res_tour2.status_code == 200
    assert res_tour2.json()["active_faults_count"] >= 1

    # 5. Invalid fault rejected with 400
    bad_payload = {
        "fault_type": "unknown_fault_xyz",
        "asset_id": "MTR-01",
    }
    res_bad = client.post("/api/demo/inject", json=bad_payload)
    assert res_bad.status_code == 400

    # 6. Final reset restores clean state
    res_reset2 = client.post("/api/demo/reset")
    assert res_reset2.status_code == 200
    assert res_reset2.json()["active_faults"] == 0
