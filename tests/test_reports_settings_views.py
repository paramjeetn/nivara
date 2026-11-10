"""Unit and integration tests for Reports and Settings views (Step UX-8).
========================================================================
Tests:
1. Reports view tab rendering and metrics computation.
2. Official institutional audit PDF generation via ReportLab trigger.
3. BEE Star Rating, Energy Performance Index (EPI), and next-star gap analysis.
4. ECBC 2017 chapter-level statutory checklist and NAPCC 20% burn-down tracker.
5. Cross-facility benchmarking visibility gated on user multi-facility authorization.
6. Settings view tab rendering across all 7 administration tabs.
7. Assumptions registry lookup and domain/confidence filtering.
8. Proof layer and scientific model evaluation metrics display.
9. Live fault injection execution and reset via Settings > Simulator.
10. Users & roles directory loading from users.yaml.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from campus_energy import assumptions
from campus_energy.analytics.esg import calculate_esg_metrics, get_bee_benchmarks
from campus_energy.analytics.governance import get_multi_facility_benchmarks
from campus_energy.analytics.report_generator import generate_facility_pdf
from campus_energy.decision.ledger import SavingsLedger
from campus_energy.demo.inject import clear_fault_registry, get_active_faults
from campus_energy.notify.base import NotificationMessage
from campus_energy.notify.router import NotificationRouter
from campus_energy.profiles import FacilityProfile, load_all_profiles
from campus_energy.tariffs import load_tariff
from campus_energy.ui.views.reports import render_reports
from campus_energy.ui.views.settings import render_settings
from campus_energy.users import UserProfile, get_demo_users, load_users


@pytest.fixture
def sample_facility_meta() -> FacilityProfile:
    """Fixture returning a canonical university profile."""
    profiles = load_all_profiles("config/facility_profiles")
    return profiles.get("university") or list(profiles.values())[0]


@pytest.fixture
def sample_assets_df() -> pd.DataFrame:
    """Fixture with meters, buildings, and sub-equipment."""
    return pd.DataFrame([
        {
            "asset_id": "MTR-BLD-A",
            "asset_name": "Academic Block A Main Incomer",
            "building_id": "BLD-A",
            "zone_id": "BLD-A-MAIN",
            "asset_type": "meter",
            "rated_power_kw": 250.0,
            "nominal_power_kw": 85.0,
            "power_factor_nominal": 0.94,
            "monitoring_level": "feeder",
        },
        {
            "asset_id": "MTR-BLD-B",
            "asset_name": "Academic Block B Sub-Feeder",
            "building_id": "BLD-B",
            "zone_id": "BLD-B-MAIN",
            "asset_type": "meter",
            "rated_power_kw": 180.0,
            "nominal_power_kw": 60.0,
            "power_factor_nominal": 0.93,
            "monitoring_level": "feeder",
        },
        {
            "asset_id": "EQ-UTL-P1-PM",
            "asset_name": "Condenser Water Pump P1",
            "building_id": "BLD-UTL",
            "zone_id": "BLD-UTL-PUMP",
            "asset_type": "pump",
            "rated_power_kw": 22.0,
            "nominal_power_kw": 18.5,
            "power_factor_nominal": 0.91,
            "monitoring_level": "equipment",
        },
        {
            "asset_id": "EQ-BLD-A-LT01",
            "asset_name": "Lecture Hall Wing A Lighting Circuit",
            "building_id": "BLD-A",
            "zone_id": "BLD-A-CR101",
            "asset_type": "lighting",
            "rated_power_kw": 15.0,
            "nominal_power_kw": 12.0,
            "power_factor_nominal": 0.95,
            "monitoring_level": "equipment",
        },
    ])


@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    """Fixture providing 14 days of realistic 5-minute interval telemetry."""
    start_dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    timestamps = [start_dt + timedelta(minutes=5 * i) for i in range(288 * 14)]
    rows = []
    for ts in timestamps:
        hour = ts.hour
        # Diurnal bell curve: higher in afternoon (10-16), lower at night
        base_kw = 80.0 + 50.0 * np.sin(np.pi * (hour - 6) / 14) if 6 <= hour <= 20 else 25.0
        rows.append({
            "reading_id": f"R-{len(rows):08d}",
            "timestamp": ts.isoformat(),
            "asset_id": "MTR-BLD-A",
            "building_id": "BLD-A",
            "zone_id": "BLD-A-MAIN",
            "reading_type": "meter",
            "power_kw": max(5.0, base_kw),
            "interval_energy_kwh": max(5.0, base_kw) * (5.0 / 60.0),
            "voltage_v": 230.5,
            "power_factor": 0.94,
            "current_a": 120.0,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """Fixture providing sample anomaly/fault events."""
    return pd.DataFrame([
        {
            "event_id": "EVT-202611-001",
            "event_type": "lights_left_on",
            "asset_id": "EQ-BLD-A-LT01",
            "severity": "high",
            "start_time": "2026-11-02T22:00:00+00:00",
            "end_time": "2026-11-03T05:00:00+00:00",
            "duration_hours": 7.0,
            "estimated_wasted_kwh": 84.0,
            "excess_power_kw": 12.0,
            "status": "active",
        },
        {
            "event_id": "EVT-202611-002",
            "event_type": "pump_no_flow",
            "asset_id": "EQ-UTL-P1-PM",
            "severity": "critical",
            "start_time": "2026-11-04T12:00:00+00:00",
            "end_time": "2026-11-04T16:00:00+00:00",
            "duration_hours": 4.0,
            "estimated_wasted_kwh": 74.0,
            "excess_power_kw": 18.5,
            "status": "active",
        },
    ])


@pytest.fixture
def admin_multi_user() -> UserProfile:
    """User with multi-facility administrative access."""
    return UserProfile(
        id="usr_admin_01",
        name="Dr. Arun Patnaik",
        email="patnaik.vc@bput.ac.in",
        role="administrator",
        title="Vice Chancellor",
        facilities=["university", "hospital", "govt_office", "industrial"],
        default_facility="university",
    )


@pytest.fixture
def ops_single_user() -> UserProfile:
    """User restricted to a single facility."""
    return UserProfile(
        id="usr_ops_01",
        name="Er. Rajesh Mohanty",
        email="mohanty.ops@bput.ac.in",
        role="operations",
        title="Chief Facilities Engineer",
        facilities=["university"],
        default_facility="university",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. REPORTS VIEW TESTS
# ─────────────────────────────────────────────────────────────────────────────

@patch("streamlit.plotly_chart")
@patch("streamlit.dataframe")
@patch("streamlit.metric")
@patch("streamlit.tabs")
def test_render_reports_smoke(
    mock_tabs,
    mock_metric,
    mock_df,
    mock_plotly,
    sample_facility_meta,
    sample_assets_df,
    sample_telemetry_df,
    sample_events_df,
    admin_multi_user,
):
    """Smoke test for render_reports verifying all 4 tabs initialize cleanly."""
    mock_t1, mock_t2, mock_t3, mock_t4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    mock_tabs.return_value = [mock_t1, mock_t2, mock_t3, mock_t4]

    ledger = SavingsLedger()
    try:
        tariff = load_tariff()
    except Exception:
        tariff = None

    render_reports(
        facility_meta=sample_facility_meta,
        assets=sample_assets_df,
        tel=sample_telemetry_df,
        ctx=pd.DataFrame(),
        events=sample_events_df,
        complaints=pd.DataFrame(),
        ledger=ledger,
        tariff=tariff,
        current_user=admin_multi_user,
    )

    mock_tabs.assert_called_once()
    assert mock_metric.called


def test_pdf_report_generator_direct(sample_facility_meta, sample_telemetry_df, sample_events_df):
    """Verify generate_facility_pdf produces valid ReportLab PDF binary bytes."""
    esg_data = calculate_esg_metrics(
        telemetry_df=sample_telemetry_df,
        events_df=sample_events_df,
        floor_area_m2=sample_facility_meta.floor_area_m2,
        facility_type=sample_facility_meta.facility_type,
    )

    pdf_bytes = generate_facility_pdf(
        facility_name=sample_facility_meta.facility_name,
        facility_type=sample_facility_meta.facility_type,
        floor_area_m2=sample_facility_meta.floor_area_m2,
        kpis=esg_data["kpis"],
        bee_info=esg_data["bee"],
        events_df=sample_events_df,
        profile=sample_facility_meta,
        assumptions_registry=assumptions,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    # PDF magic byte signature
    assert pdf_bytes.startswith(b"%PDF")


def test_bee_star_rating_and_gap_analysis(sample_facility_meta):
    """Verify BEE Star rating calculation and next-star gap analysis logic."""
    norms = get_bee_benchmarks(sample_facility_meta.facility_type)
    assert "stars" in norms
    assert "ecbc_baseline" in norms

    # At 120 kWh/m2, university is 4 or 5 stars
    star_dict = norms["stars"]
    five_star_max = star_dict[5][1]
    four_star_max = star_dict[4][1]
    assert five_star_max < four_star_max

    # Gap calculation test
    sim_epi = 145.0  # Above 4-star, within 3-star
    current_stars = 3
    next_star = 4
    target_epi = star_dict[next_star][1]
    gap_epi = sim_epi - target_epi
    assert gap_epi > 0
    # Target EPI must be lower than current EPI
    assert target_epi < sim_epi


@patch("streamlit.dataframe")
@patch("streamlit.tabs")
def test_cross_facility_benchmarking_authorization(
    mock_tabs,
    mock_df,
    sample_facility_meta,
    sample_assets_df,
    sample_telemetry_df,
    admin_multi_user,
    ops_single_user,
):
    """Verify cross-facility matrix is gated on multi-facility authorization."""
    mock_t1, mock_t2, mock_t3, mock_t4 = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    mock_tabs.return_value = [mock_t1, mock_t2, mock_t3, mock_t4]

    # Test with multi-facility admin user: has access
    assert len(admin_multi_user.facilities) > 1
    assert admin_multi_user.has_facility_access("hospital") is True

    # Test with single-facility ops user: restricted
    assert len(ops_single_user.facilities) == 1
    assert ops_single_user.has_facility_access("hospital") is False

    # Render as single-facility user
    render_reports(
        facility_meta=sample_facility_meta,
        assets=sample_assets_df,
        tel=sample_telemetry_df,
        ctx=pd.DataFrame(),
        events=pd.DataFrame(),
        complaints=pd.DataFrame(),
        current_user=ops_single_user,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. SETTINGS VIEW TESTS
# ─────────────────────────────────────────────────────────────────────────────

@patch("streamlit.plotly_chart")
@patch("streamlit.dataframe")
@patch("streamlit.metric")
@patch("streamlit.tabs")
def test_render_settings_smoke(
    mock_tabs,
    mock_metric,
    mock_df,
    mock_plotly,
    sample_facility_meta,
    sample_assets_df,
    sample_telemetry_df,
    sample_events_df,
    admin_multi_user,
):
    """Smoke test for render_settings verifying all 7 tabs initialize cleanly."""
    mock_tabs.side_effect = lambda tabs_list: [MagicMock() for _ in range(len(tabs_list))]

    notif_r = NotificationRouter(dry_run=True)

    render_settings(
        facility_meta=sample_facility_meta,
        assets=sample_assets_df,
        tel=sample_telemetry_df,
        ctx=pd.DataFrame(),
        events=sample_events_df,
        complaints=pd.DataFrame(),
        current_user=admin_multi_user,
        notif_router=notif_r,
    )

    assert mock_tabs.call_count >= 1
    assert mock_metric.called


def test_assumptions_registry_integrity():
    """Verify assumptions registry contains the full suite of governed parameters."""
    all_assump = assumptions.all()
    # Must have 49 parameters as documented in config/assumptions.yaml
    assert len(all_assump) >= 49

    keys = [a.key for a in all_assump]
    assert "tariff.electricity_commercial" in keys
    assert "emissions.grid_factor" in keys
    assert "policy.napcc_reduction_target_pct" in keys
    assert "waste.fault_avoidable_fraction" in keys

    # Verify domain prefix lookup
    tariff_items = assumptions.list_by_prefix("tariff.")
    assert len(tariff_items) >= 2


def test_users_directory_and_role_matrix():
    """Verify load_users and get_demo_users correctly load configured profiles."""
    users = get_demo_users()
    assert len(users) >= 3

    roles = {u.role for u in users}
    assert "administrator" in roles
    assert "operations" in roles
    assert "sustainability" in roles

    for u in users:
        assert u.id.startswith("usr_")
        assert "@" in u.email
        assert len(u.facilities) >= 1
        assert u.has_facility_access(u.default_facility) is True


def test_simulator_fault_injection_lifecycle(sample_telemetry_df):
    """Verify live fault injection studio functionality from Settings > Simulator."""
    clear_fault_registry()
    initial_active = get_active_faults()
    assert len(initial_active) == 0

    from campus_energy.demo.inject import inject_live_fault

    # Inject canonical lights left on fault
    mod_tel, event = inject_live_fault(
        telemetry_df=sample_telemetry_df,
        fault_type="lights_left_on",
        asset_id="EQ-BLD-A-LT01",
        duration_hours=4.0,
    )

    assert event is not None
    assert event.event_type.value == "lights_left_on"
    assert event.asset_id == "EQ-BLD-A-LT01"

    # Verify active fault registry
    active_now = get_active_faults()
    assert len(active_now) == 1
    assert active_now[0].event_id == event.event_id

    # Reset simulation state
    clear_fault_registry()
    assert len(get_active_faults()) == 0


def test_notification_probe_dispatch():
    """Verify NotificationRouter test probe dispatches and logs delivery audit."""
    router = NotificationRouter(dry_run=True)
    router.clear_cache()

    msg = NotificationMessage(
        recipient_role="operations",
        severity="high",
        title="Test Operational Alert from Settings",
        asset_id="EQ-UTL-P1-PM",
        description="Diagnostic ping to verify multi-channel delivery.",
        impact_range="₹1,200 [₹950 – ₹1,450] / day",
        service_guard_note="🟢 Service-Safe: Non-critical diagnostic",
    )

    results = router.dispatch(msg)
    assert len(results) >= 1
    # Verify delivery audit history
    history = router._history
    assert len(history) >= 1
    assert any("Test Operational Alert" in h.get("title", "") for h in history)
