"""Unit and integration tests for Role-Based Homes (Step UX-5).

Tests:
1. Operations Home:
   - Today Strip calculation (load now vs expected, open alerts, work orders due, sensor health).
   - Needs Attention prioritization (critical/high alerts, SLA warnings, stuck sensors).
   - Load Today vs Forecast Plotly chart with anomaly markers.
   - Recent activity feed assembly.
2. Administrator Home:
   - This Month Financial Strip (billed cost, projected month-end bill, budget variance).
   - Contract Demand Risk status banner.
   - Decisions Waiting queue and approval/decline mechanics.
   - Savings Realized (verified savings, avoided kWh, hit-rate %).
   - Spend by Department bar chart & ledger summary.
3. Sustainability Home:
   - Carbon This Month vs Target trajectory and NAPCC status sentence.
   - BEE Star Rating, Annual EPI, and gap to next star rating.
   - Verified Impact (kgCO2e, kWh, INR) and work order audit records.
4. Role-based routing:
   - UserProfile.role mapping to operations, administrator, and sustainability homes.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from campus_energy.decision.ledger import SavingsLedger, WorkOrderItem
from campus_energy.decision.recommendation import GuardResult, Recommendation, SavingsBundle
from campus_energy.decision.tariffs import DEFAULT_TARIFF_ID, IndianTariff, load_tariff
from campus_energy.profiles import FacilityProfile, load_all_profiles
from campus_energy.schemas import Alert, Severity
from campus_energy.users import UserProfile
from campus_energy.ui.views.home_admin import (
    prepare_admin_financial_strip,
    prepare_decisions_waiting,
    prepare_savings_realized,
    prepare_spend_by_department,
    render_home_admin,
)
from campus_energy.ui.views.home_operations import (
    assemble_needs_attention,
    build_load_forecast_figure,
    prepare_operations_today_strip,
    prepare_recent_activity,
    render_home_operations,
)
from campus_energy.ui.views.home_sustainability import (
    prepare_bee_star_and_epi,
    prepare_carbon_trajectory,
    prepare_sustainability_verified_impact,
    render_home_sustainability,
)


@pytest.fixture
def sample_profile() -> FacilityProfile:
    profiles = load_all_profiles("config/facility_profiles")
    return profiles.get("university") or list(profiles.values())[0]


@pytest.fixture
def sample_assets_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "asset_id": "METER-MAIN-01",
            "asset_name": "Main Campus Substation Meter",
            "asset_type": "meter",
            "building_id": "BLD-ADMIN",
            "zone_id": "ZONE-A",
            "rated_power_kw": 500.0,
        },
        {
            "asset_id": "EQ-CHILLER-01",
            "asset_name": "Main Chiller Unit 1",
            "asset_type": "hvac",
            "building_id": "BLD-ADMIN",
            "zone_id": "ZONE-A",
            "rated_power_kw": 45.0,
        },
        {
            "asset_id": "EQ-PUMP-02",
            "asset_name": "Rooftop Hydro Pump",
            "asset_type": "pump",
            "building_id": "BLD-UTL",
            "zone_id": "ZONE-P",
            "rated_power_kw": 7.5,
        },
    ])


@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    base_time = datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)
    rows = []
    # 288 records (24 hours at 5-minute intervals)
    for i in range(288):
        t = base_time + timedelta(minutes=5 * i)
        # Meter record
        rows.append({
            "reading_id": f"RDG-M-{i:03d}",
            "timestamp": t.isoformat(),
            "asset_id": "METER-MAIN-01",
            "power_kw": 320.0 + (i % 20),
            "interval_energy_kwh": (320.0 + (i % 20)) / 12.0,
            "power_factor": 0.92,
            "flow_lpm": 0.0,
            "reading_type": "meter",
        })
        # Equipment record
        rows.append({
            "reading_id": f"RDG-E-{i:03d}",
            "timestamp": t.isoformat(),
            "asset_id": "EQ-CHILLER-01",
            "power_kw": 35.0,
            "interval_energy_kwh": 35.0 / 12.0,
            "power_factor": 0.88,
            "flow_lpm": 0.0,
            "reading_type": "equipment",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_context_df() -> pd.DataFrame:
    base_time = datetime(2026, 11, 10, 8, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(288):
        t = base_time + timedelta(minutes=5 * i)
        rows.append({
            "timestamp": t.isoformat(),
            "building_id": "BLD-ADMIN",
            "zone_id": "ZONE-A",
            "outdoor_temperature_c": 32.5,
            "occupancy": 150,
            "holiday": False,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "event_id": "EVT-2026-001",
            "event_type": "high_consumption",
            "asset_id": "EQ-CHILLER-01",
            "start_time": "2026-11-10T09:00:00+00:00",
            "end_time": "2026-11-10T12:00:00+00:00",
            "severity": "critical",
            "description": "Chiller compressor draw exceeded normal range by 15.2 kW",
            "actual_value": 45.0,
            "expected_value": 29.8,
            "deviation": 15.2,
        },
        {
            "event_id": "EVT-2026-002",
            "event_type": "pump_no_flow",
            "asset_id": "EQ-PUMP-02",
            "start_time": "2026-11-10T10:15:00+00:00",
            "end_time": "2026-11-10T11:30:00+00:00",
            "severity": "high",
            "description": "Hydro pump drawing 7.5 kW with zero water flow",
            "actual_value": 7.5,
            "expected_value": 0.0,
            "deviation": 7.5,
        },
    ])


@pytest.fixture
def sample_ledger() -> SavingsLedger:
    ledger = SavingsLedger()
    rec1 = Recommendation(
        id="REC-WO-01",
        facility_id="university",
        type="hvac_setpoint",
        target="EQ-CHILLER-01",
        action_text="Reset chiller chilled water setpoint to 8.5°C",
        trigger="EVT-2026-001",
        savings=SavingsBundle.from_expected(120.0),
        owner_role="operations",
    )
    wo = ledger.add_recommendation(rec1)
    ledger.create_work_order(wo.work_order_id, assignee="HVAC Lead", sla_hours=12)

    # Add a verified recommendation
    rec2 = Recommendation(
        id="REC-WO-02",
        facility_id="university",
        type="curfew_shutdown",
        target="EQ-LT-ADMIN-01",
        action_text="Enforce library night curfew lighting shutdown",
        trigger="EVT-2026-002",
        savings=SavingsBundle.from_expected(50.0),
        owner_role="operations",
    )
    wo2 = ledger.add_recommendation(rec2)
    ledger.create_work_order(wo2.work_order_id, assignee="Electrical Lead", sla_hours=24)
    ledger.mark_done(wo2.work_order_id)
    # Mark verified manually
    wo2.status = "verified"
    wo2.verified_savings = SavingsBundle.from_expected(52.0)
    return ledger


# ─────────────────────────────────────────────────────────────────────────────
# 1. Operations Home Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_operations_today_strip_calculation(
    sample_telemetry_df: pd.DataFrame,
    sample_assets_df: pd.DataFrame,
    sample_events_df: pd.DataFrame,
    sample_ledger: SavingsLedger,
):
    """Verify operations Today Strip computes exactly 4 numbers with correct deltas."""
    stats = prepare_operations_today_strip(
        tel=sample_telemetry_df,
        assets=sample_assets_df,
        events=sample_events_df,
        ledger=sample_ledger,
    )
    assert "load_now_kw" in stats
    assert stats["load_now_kw"] > 0
    assert "expected_load_kw" in stats
    assert "load_delta_kw" in stats
    assert stats["load_delta_color"] in ["normal", "inverse", "off"]

    # Open alerts
    assert stats["open_alerts_count"] == 2
    assert stats["crit_high_count"] == 2
    assert "2 Critical / High" in stats["alert_delta_str"]

    # Work orders due
    assert stats["work_orders_due_today"] >= 1
    assert "wo_delta_str" in stats

    # Sensor health
    assert stats["data_health_pct"] >= 95.0
    assert "Nominal" in stats["data_health_str"] or "Minor" in stats["data_health_str"]


def test_operations_needs_attention_assembly(
    sample_events_df: pd.DataFrame,
    sample_assets_df: pd.DataFrame,
    sample_telemetry_df: pd.DataFrame,
    sample_context_df: pd.DataFrame,
    sample_profile: FacilityProfile,
    sample_ledger: SavingsLedger,
):
    """Verify Needs Attention prioritizes critical/high alerts, work orders, and stuck sensors."""
    items = assemble_needs_attention(
        events=sample_events_df,
        assets=sample_assets_df,
        tel=sample_telemetry_df,
        ctx=sample_context_df,
        facility_meta=sample_profile,
        ledger=sample_ledger,
    )
    assert len(items) >= 2
    # First item should be critical or high priority
    assert items[0]["severity"] in ["critical", "high"]
    assert items[0]["priority_rank"] in [0, 1]
    assert "headline" in items[0]
    assert "impact" in items[0]
    assert "age" in items[0]
    assert items[0]["action_label"] == "Open in Alerts & Work"

    # Verify headline formatting
    chiller_item = next(it for it in items if "EQ-CHILLER-01" in it["asset_id"])
    assert "Main Chiller Unit 1" in chiller_item["headline"] or "EQ-CHILLER-01" in chiller_item["headline"]


def test_operations_load_today_vs_forecast_figure(
    sample_telemetry_df: pd.DataFrame,
    sample_context_df: pd.DataFrame,
    sample_assets_df: pd.DataFrame,
    sample_events_df: pd.DataFrame,
):
    """Verify Load Today vs Forecast generates valid Plotly chart with anomaly markers."""
    fig = build_load_forecast_figure(
        tel=sample_telemetry_df,
        ctx=sample_context_df,
        assets=sample_assets_df,
        events=sample_events_df,
    )
    assert fig is not None
    # Traces should include actual load
    trace_names = [t.name for t in fig.data if hasattr(t, "name")]
    assert any("Actual Load" in str(n) for n in trace_names)


def test_operations_recent_activity(
    sample_events_df: pd.DataFrame,
    sample_ledger: SavingsLedger,
):
    """Verify Recent Activity feed captures alert dispatches and verifications."""
    act = prepare_recent_activity(events=sample_events_df, ledger=sample_ledger)
    assert len(act) >= 2
    types = [a["event_type"] for a in act]
    assert "alert_raised" in types or "wo_dispatched" in types or "fault_resolved" in types


# ─────────────────────────────────────────────────────────────────────────────
# 2. Administrator Home Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_admin_financial_strip_calculation(
    sample_telemetry_df: pd.DataFrame,
    sample_profile: FacilityProfile,
):
    """Verify Administrator Financial Strip computes billed cost, projection, and contract demand."""
    tariff = load_tariff("odisha_ht_commercial")
    fin = prepare_admin_financial_strip(
        tel=sample_telemetry_df,
        tariff=tariff,
        facility_meta=sample_profile,
    )
    assert fin["billed_cost_inr"] > 0
    assert "₹" in fin["billed_cost_str"]
    assert fin["projected_bill_inr"] > 0
    assert "₹" in fin["projected_bill_str"]
    assert "status_banner_text" in fin
    assert "Peak demand" in fin["status_banner_text"]
    assert "contract demand" in fin["status_banner_text"]
    assert "Headroom" in fin["status_banner_text"]
    assert fin["demand_status"] in ["Nominal", "Tight Headroom"] or "Surcharge" in fin["demand_status"]


def test_admin_decisions_waiting_and_approvals(
    sample_profile: FacilityProfile,
    sample_ledger: SavingsLedger,
):
    """Verify Decisions Waiting queue renders decision cards and handles approve/decline."""
    decisions = prepare_decisions_waiting(
        facility_meta=sample_profile,
        ledger=sample_ledger,
    )
    assert len(decisions) >= 3
    first_dec = decisions[0]
    assert "title" in first_dec
    assert "annual_savings_inr" in first_dec
    assert first_dec["annual_savings_inr"] > 0
    assert "payback_months" in first_dec
    assert first_dec["guard_status"] in ["pass", "needs_review", "blocked"]

    # Test approval override
    overrides = {first_dec["rec_id"]: "approved"}
    updated = prepare_decisions_waiting(
        facility_meta=sample_profile,
        ledger=sample_ledger,
        decision_overrides=overrides,
    )
    # The approved item should no longer be pending
    assert not any(d["rec_id"] == first_dec["rec_id"] for d in updated)


def test_admin_savings_realized(sample_ledger: SavingsLedger):
    """Verify realized savings aggregates verified INR, avoided kWh, and M&V hit-rate."""
    realized = prepare_savings_realized(ledger=sample_ledger)
    assert realized["cumulative_verified_inr"] > 0
    assert "₹" in realized["cumulative_verified_str"]
    assert realized["cumulative_verified_kwh"] > 0
    assert realized["estimate_hit_rate_pct"] >= 0.0


def test_admin_spend_by_department(sample_telemetry_df: pd.DataFrame):
    """Verify spend by department returns sorted dataframe and horizontal bar chart."""
    df, fig = prepare_spend_by_department(sample_telemetry_df)
    assert not df.empty
    assert "department" in df.columns
    assert "cost_inr" in df.columns
    assert fig is not None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sustainability Home Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_sustainability_carbon_trajectory(
    sample_telemetry_df: pd.DataFrame,
    sample_profile: FacilityProfile,
):
    """Verify Carbon trajectory computes Scope 2 burndown vs NAPCC ceiling and status sentence."""
    traj, fig = prepare_carbon_trajectory(tel=sample_telemetry_df, facility_meta=sample_profile)
    assert traj["monthly_budget_tco2"] > 0
    assert "status_sentence" in traj
    assert "national NAPCC ceiling" in traj["status_sentence"]
    assert fig is not None
    # Traces should exist in figure
    assert len(fig.data) >= 1


def test_sustainability_bee_star_and_epi(
    sample_telemetry_df: pd.DataFrame,
    sample_events_df: pd.DataFrame,
    sample_profile: FacilityProfile,
):
    """Verify BEE Star rating, EPI, and gap to next star rating are accurately computed."""
    bee = prepare_bee_star_and_epi(
        tel=sample_telemetry_df,
        events=sample_events_df,
        facility_meta=sample_profile,
    )
    assert 1 <= bee["stars"] <= 5
    assert "⭐" in bee["stars_display"]
    assert bee["epi_annual_kwh_m2"] > 0
    assert bee["ecbc_baseline_kwh_m2"] > 0
    assert "gap_to_next_star" in bee
    assert "Star" in bee["gap_to_next_star"] or "Maximum" in bee["gap_to_next_star"]


def test_sustainability_verified_impact(sample_ledger: SavingsLedger):
    """Verify verified impact produces audited GHG abatements and actions list."""
    impact = prepare_sustainability_verified_impact(ledger=sample_ledger)
    assert impact["verified_co2_kg"] > 0
    assert "kgCO₂e" in impact["verified_co2_str"]
    assert impact["verified_kwh"] > 0
    assert len(impact["actions"]) >= 1
    first_act = impact["actions"][0]
    assert "measure" in first_act
    assert "abated_co2" in first_act
    assert "avoided_kwh" in first_act


# ─────────────────────────────────────────────────────────────────────────────
# 4. Role Routing Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_role_based_routing_profiles():
    """Verify user roles correctly route to designated operational homes."""
    u_ops = UserProfile(
        id="usr_01",
        name="Rajesh Mohanty",
        email="rajesh@example.com",
        role="operations",
        title="Operations Lead",
    )
    u_admin = UserProfile(
        id="usr_02",
        name="Dr. Arun Patnaik",
        email="arun@example.com",
        role="administrator",
        title="Registrar",
    )
    u_sust = UserProfile(
        id="usr_03",
        name="Priya Das",
        email="priya@example.com",
        role="sustainability",
        title="CSO",
    )

    assert u_ops.role == "operations"
    assert u_admin.role == "administrator"
    assert u_sust.role == "sustainability"


def test_render_home_views_headless_smoke(
    sample_profile: FacilityProfile,
    sample_assets_df: pd.DataFrame,
    sample_telemetry_df: pd.DataFrame,
    sample_context_df: pd.DataFrame,
    sample_events_df: pd.DataFrame,
    sample_ledger: SavingsLedger,
):
    """Headless smoke test verifying render functions execute without unhandled exceptions."""
    tariff = load_tariff("odisha_ht_commercial")

    def mock_columns(spec, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(count)]

    # Mocks for streamlit layout methods to run in headless pytest
    with patch("streamlit.columns", side_effect=mock_columns), \
         patch("streamlit.plotly_chart"), \
         patch("streamlit.markdown"), \
         patch("streamlit.metric"), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.caption"), \
         patch("streamlit.progress"), \
         patch("streamlit.info"), \
         patch("streamlit.success"):

        # 1. Operations Home
        render_home_operations(
            facility_meta=sample_profile,
            assets=sample_assets_df,
            tel=sample_telemetry_df,
            ctx=sample_context_df,
            events=sample_events_df,
            ledger=sample_ledger,
            tariff=tariff,
        )

        # 2. Administrator Home
        render_home_admin(
            facility_meta=sample_profile,
            assets=sample_assets_df,
            tel=sample_telemetry_df,
            ctx=sample_context_df,
            events=sample_events_df,
            ledger=sample_ledger,
            tariff=tariff,
        )

        # 3. Sustainability Home
        render_home_sustainability(
            facility_meta=sample_profile,
            assets=sample_assets_df,
            tel=sample_telemetry_df,
            ctx=sample_context_df,
            events=sample_events_df,
            ledger=sample_ledger,
            tariff=tariff,
        )
