"""Unit and integration tests for Step UX-9: Ask Nivara Slide-Over Panel.

Verifies:
- Context detection from Asset Detail Page (asset_id in state or query params).
- Context detection from Alert Detail Drawer (alert_id in state or query params).
- Context detection for Role Homes and operational views (Home, Energy, Water, Reports, Settings).
- Contextual quick prompt generation tailored to active screen and role persona.
- Offline fallback intelligence response generation with asset and alert grounding.
- Action parsing, recommendation synthesis, and Service-Impact Guard evaluation.
- Graceful rendering of Ask Nivara slide-over panel.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import streamlit as st

from campus_energy.decision.guard import check_service_guard
from campus_energy.decision.ledger import SavingsLedger
from campus_energy.profiles import FacilityProfile, ProtectedLoad, load_facility_profile
from campus_energy.users import UserProfile
from campus_energy.ui.panels.ask_nivara import (
    evaluate_actions_with_guard,
    get_contextual_quick_prompts,
    render_ask_nivara_panel,
    resolve_ask_nivara_context,
)
from campus_energy.dashboard.agent import (
    actions_to_recommendations,
    chat_with_gemini,
    generate_offline_fallback,
    parse_actions_from_response,
)


@pytest.fixture
def mock_facility_profile() -> FacilityProfile:
    """Fixture providing a test facility profile with protected loads."""
    profile = load_facility_profile("config/facility_profiles/university.yaml")
    p_load = ProtectedLoad(
        asset_id="EQ-CRIT-ICU-01",
        asset_name="ICU Ventilator Bank & Chilled Water Loop",
        category="life_safety",
        building_id="BLD-HOSP-01",
        zone_id="ZONE-ICU",
        reason="Clinical critical care patient life support",
    )
    profile.protected_loads.append(p_load)
    return profile


@pytest.fixture
def mock_telemetry_dfs():
    """Fixture providing test assets, telemetry, and events dataframes."""
    assets = pd.DataFrame([
        {
            "asset_id": "EQ-UTL-P1-PM",
            "asset_name": "Utility Water Pump 1",
            "equipment_type": "Water Pump",
            "building_id": "BLD-UTL-01",
            "zone_id": "PUMP-RM",
        },
        {
            "asset_id": "EQ-CRIT-ICU-01",
            "asset_name": "ICU Ventilator Chiller",
            "equipment_type": "Chiller",
            "building_id": "BLD-HOSP-01",
            "zone_id": "ZONE-ICU",
        },
        {
            "asset_id": "BLD-101-HVAC-01",
            "asset_name": "Admin Academic AHU 1",
            "equipment_type": "HVAC AHU",
            "building_id": "BLD-101",
            "zone_id": "FLOOR-1",
        },
    ])

    tel = pd.DataFrame([
        {"timestamp": "2026-11-10T08:00:00Z", "asset_id": "EQ-UTL-P1-PM", "power_kw": 22.4, "baseline_kw": 18.0},
        {"timestamp": "2026-11-10T08:05:00Z", "asset_id": "EQ-UTL-P1-PM", "power_kw": 25.1, "baseline_kw": 18.0},
        {"timestamp": "2026-11-10T08:00:00Z", "asset_id": "EQ-CRIT-ICU-01", "power_kw": 48.0, "baseline_kw": 45.0},
        {"timestamp": "2026-11-10T08:05:00Z", "asset_id": "EQ-CRIT-ICU-01", "power_kw": 49.5, "baseline_kw": 45.0},
        {"timestamp": "2026-11-10T08:05:00Z", "asset_id": "BLD-101-HVAC-01", "power_kw": 62.0, "baseline_kw": 40.0},
    ])

    events = pd.DataFrame([
        {
            "event_id": "EVT-001",
            "alert_id": "ALT-001",
            "event_type": "Overnight Lighting Spike",
            "asset_id": "BLD-101-HVAC-01",
            "deviation": 22.0,
            "excess_kw": 22.0,
            "description": "Triple-engine consensus: Occupancy schedule override",
        },
        {
            "event_id": "EVT-002",
            "alert_id": "ALT-002",
            "event_type": "Critical Hospital Chiller Surge",
            "asset_id": "EQ-CRIT-ICU-01",
            "deviation": 15.0,
            "excess_kw": 15.0,
            "description": "ICU thermal surge detected during peak hours",
        },
    ])

    complaints = pd.DataFrame([
        {"timestamp": "2026-11-10T08:00:00Z", "description": "2nd floor too warm", "category": "thermal"},
    ])

    return assets, tel, events, complaints


# ─────────────────────────────────────────────────────────────────────────────
# 1. Context Resolution Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_context_resolution_asset_detail_page(mock_facility_profile, mock_telemetry_dfs):
    """Test context resolution when an asset detail page is active."""
    assets, tel, events, complaints = mock_telemetry_dfs
    user = UserProfile(id="u1", name="Arun Patel", role="operations", email="arun@campus.ac.in", title="Operations Lead", facilities=["*"])

    # Simulate Asset Detail Page active in session state for non-protected asset
    st.session_state["active_asset_detail_id"] = "BLD-101-HVAC-01"
    st.session_state.pop("active_alert_drawer_id", None)
    if "alert_id" in st.query_params:
        del st.query_params["alert_id"]

    ctx = resolve_ask_nivara_context(
        facility_meta=mock_facility_profile,
        assets_df=assets,
        tel_df=tel,
        events_df=events,
        current_user=user,
        complaints_df=complaints,
    )

    assert ctx["context_type"] == "asset"
    assert ctx["asset_id"] == "BLD-101-HVAC-01"
    assert ctx["asset_name"] == "Admin Academic AHU 1"
    assert "BLD-101-HVAC-01" in ctx["chip_label"]
    assert ctx["current_kw"] == 62.0
    assert ctx["baseline_delta"] == 22.0
    assert ctx["is_protected"] is False
    assert "Current Power Draw: 62.0 kW" in ctx["prompt_context"]


def test_context_resolution_asset_protected(mock_facility_profile, mock_telemetry_dfs):
    """Test context resolution for a mission-critical protected asset."""
    assets, tel, events, complaints = mock_telemetry_dfs
    user = UserProfile(id="u1", name="Arun Patel", role="operations", email="arun@campus.ac.in", title="Operations Lead", facilities=["*"])

    st.session_state["active_asset_detail_id"] = "EQ-CRIT-ICU-01"
    st.session_state.pop("active_alert_drawer_id", None)

    ctx = resolve_ask_nivara_context(
        facility_meta=mock_facility_profile,
        assets_df=assets,
        tel_df=tel,
        events_df=events,
        current_user=user,
    )

    assert ctx["context_type"] == "asset"
    assert ctx["asset_id"] == "EQ-CRIT-ICU-01"
    assert ctx["is_protected"] is True
    assert "PROTECTED" in ctx["prompt_context"]


def test_context_resolution_alert_drawer(mock_facility_profile, mock_telemetry_dfs):
    """Test context resolution when an alert detail drawer is active."""
    assets, tel, events, complaints = mock_telemetry_dfs
    user = UserProfile(id="u1", name="Arun Patel", role="operations", email="arun@campus.ac.in", title="Operations Lead", facilities=["*"])

    st.session_state["active_alert_drawer_id"] = "ALT-001"
    st.session_state.pop("active_asset_detail_id", None)

    ctx = resolve_ask_nivara_context(
        facility_meta=mock_facility_profile,
        assets_df=assets,
        tel_df=tel,
        events_df=events,
        current_user=user,
    )

    assert ctx["context_type"] == "alert"
    assert ctx["alert_id"] == "ALT-001"
    assert "Alert ALT-001" in ctx["chip_label"]
    assert ctx["excess_kw"] == 22.0
    assert "Occupancy schedule override" in ctx["factor_attribution"]
    assert "Pass" in ctx["service_guard_status"]


def test_context_resolution_alert_drawer_protected(mock_facility_profile, mock_telemetry_dfs):
    """Test context resolution for an alert on a protected mission-critical asset."""
    assets, tel, events, complaints = mock_telemetry_dfs
    user = UserProfile(id="u1", name="Arun Patel", role="operations", email="arun@campus.ac.in", title="Operations Lead", facilities=["*"])

    st.session_state["active_alert_drawer_id"] = "ALT-002"
    st.session_state.pop("active_asset_detail_id", None)

    ctx = resolve_ask_nivara_context(
        facility_meta=mock_facility_profile,
        assets_df=assets,
        tel_df=tel,
        events_df=events,
        current_user=user,
    )

    assert ctx["context_type"] == "alert"
    assert ctx["is_protected"] is True
    assert "Blocked" in ctx["service_guard_status"]


def test_context_resolution_role_and_view(mock_facility_profile, mock_telemetry_dfs):
    """Test context resolution for role homes and non-drawer operational views."""
    assets, tel, events, complaints = mock_telemetry_dfs

    # Clean drawer state
    st.session_state.pop("active_alert_drawer_id", None)
    st.session_state.pop("active_asset_detail_id", None)
    if "alert_id" in st.query_params:
        del st.query_params["alert_id"]
    if "asset" in st.query_params:
        del st.query_params["asset"]

    # Administrator on Home
    admin_user = UserProfile(id="adm", name="Dean Mishra", role="administrator", email="dean@campus.ac.in", title="Dean of Administration", facilities=["*"])
    st.session_state["selected_nav_destination"] = "Home"
    ctx_admin = resolve_ask_nivara_context(facility_meta=mock_facility_profile, current_user=admin_user)
    assert ctx_admin["context_type"] == "role_view"
    assert "Administrator" in ctx_admin["chip_label"]

    # Operations on Energy view
    st.session_state["selected_nav_destination"] = "Energy"
    ctx_energy = resolve_ask_nivara_context(facility_meta=mock_facility_profile, current_user=admin_user)
    assert "Energy Monitoring" in ctx_energy["chip_label"]

    # Sustainability on Reports view
    st.session_state["selected_nav_destination"] = "Reports"
    ctx_rep = resolve_ask_nivara_context(facility_meta=mock_facility_profile, current_user=admin_user)
    assert "Statutory Reports" in ctx_rep["chip_label"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Contextual Quick Prompts Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_contextual_quick_prompts_asset():
    """Test prompt suggestions for asset context."""
    ctx = {"context_type": "asset", "asset_id": "EQ-UTL-P1-PM"}
    prompts = get_contextual_quick_prompts(ctx, user_role="operations")

    assert len(prompts) == 4
    assert any("fluctuations" in p.lower() for p in prompts)
    assert any("protected from curtailment" in p.lower() for p in prompts)
    assert any("peer units" in p.lower() for p in prompts)


def test_contextual_quick_prompts_alert():
    """Test prompt suggestions for alert context."""
    ctx = {"context_type": "alert", "alert_id": "ALT-001"}
    prompts = get_contextual_quick_prompts(ctx, user_role="operations")

    assert len(prompts) == 4
    assert any("shut off this asset safely" in p.lower() for p in prompts)
    assert any("root cause" in p.lower() for p in prompts)


def test_contextual_quick_prompts_role_tailoring():
    """Test prompt suggestions for different roles in general view."""
    ctx = {"context_type": "role_view"}

    # Administrator
    p_admin = get_contextual_quick_prompts(ctx, user_role="administrator")
    assert any("peak demand charge risk" in p.lower() for p in p_admin)

    # Operations
    p_ops = get_contextual_quick_prompts(ctx, user_role="operations")
    assert any("work orders are breaching sla" in p.lower() for p in p_ops)

    # Sustainability
    p_sust = get_contextual_quick_prompts(ctx, user_role="sustainability")
    assert any("bee star rating" in p.lower() for p in p_sust)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Grounded Offline Fallback with Context
# ─────────────────────────────────────────────────────────────────────────────

def test_offline_fallback_asset_context_fluctuations(mock_facility_profile, mock_telemetry_dfs):
    """Test offline fallback response when querying power fluctuations for an asset."""
    assets, tel, events, complaints = mock_telemetry_dfs
    asset_ctx = {
        "context_type": "asset",
        "asset_id": "EQ-UTL-P1-PM",
        "asset_name": "Utility Water Pump 1",
        "equipment_type": "Water Pump",
        "building_id": "BLD-UTL-01",
        "current_kw": 25.1,
        "baseline_delta": 7.1,
        "is_protected": False,
        "chip_label": "🎯 Context: Asset EQ-UTL-P1-PM (Utility Water Pump 1)",
    }

    resp = chat_with_gemini(
        api_key=None,
        messages=[{"role": "user", "content": "Explain recent power fluctuations for this asset."}],
        events_df=events,
        assets_df=assets,
        complaints_df=complaints,
        active_role="operations",
        profile=mock_facility_profile,
        context_data=asset_ctx,
    )

    assert "Asset Telemetry Analysis" in resp
    assert "EQ-UTL-P1-PM" in resp
    assert "25.1 kW" in resp
    assert "7.1 kW" in resp
    assert "Offline Facility Intelligence Mode" in resp

    actions = parse_actions_from_response(resp)
    assert len(actions) >= 1


def test_offline_fallback_alert_curtailment_safety(mock_facility_profile, mock_telemetry_dfs):
    """Test safety evaluation for an alert on a protected load vs non-protected load."""
    assets, tel, events, complaints = mock_telemetry_dfs

    # 1. Protected load alert
    prot_alert_ctx = {
        "context_type": "alert",
        "alert_id": "ALT-002",
        "asset_id": "EQ-CRIT-ICU-01",
        "anomaly_type": "Critical Hospital Chiller Surge",
        "excess_kw": 15.0,
        "is_protected": True,
        "factor_attribution": "ICU thermal surge",
    }
    resp_prot = chat_with_gemini(
        api_key=None,
        messages=[{"role": "user", "content": "Can we shut off this asset safely right now?"}],
        events_df=events,
        assets_df=assets,
        complaints_df=complaints,
        active_role="operations",
        profile=mock_facility_profile,
        context_data=prot_alert_ctx,
    )
    assert "BLOCKED" in resp_prot or "Prohibited" in resp_prot
    assert "Mission-Critical" in resp_prot

    # 2. Non-protected load alert
    safe_alert_ctx = {
        "context_type": "alert",
        "alert_id": "ALT-001",
        "asset_id": "BLD-101-HVAC-01",
        "anomaly_type": "Overnight Lighting Spike",
        "excess_kw": 22.0,
        "is_protected": False,
        "factor_attribution": "Occupancy override",
    }
    resp_safe = chat_with_gemini(
        api_key=None,
        messages=[{"role": "user", "content": "Can we shut off this asset safely right now?"}],
        events_df=events,
        assets_df=assets,
        complaints_df=complaints,
        active_role="operations",
        profile=mock_facility_profile,
        context_data=safe_alert_ctx,
    )
    assert "SAFE TO CURTAIL" in resp_safe or "Curtailment Approved" in resp_safe


# ─────────────────────────────────────────────────────────────────────────────
# 4. Action Extraction and Service-Impact Guard Integration
# ─────────────────────────────────────────────────────────────────────────────

def test_action_extraction_and_service_guard_evaluation(mock_facility_profile):
    """Test action parsing, recommendation translation, and Service Guard check."""
    actions = [
        {
            "action_type": "Equipment Shutdown",
            "target_asset": "EQ-CRIT-ICU-01",  # Protected ICU load!
            "suggested_time": "Immediately",
            "estimated_saving_kwh": 30.0,
        },
        {
            "action_type": "HVAC Inspection/Schedule",
            "target_asset": "BLD-101",          # Standard non-protected
            "suggested_time": "Immediately",
            "estimated_saving_kwh": 40.0,
        },
    ]

    evaluated = evaluate_actions_with_guard(actions, mock_facility_profile)
    assert len(evaluated) == 2

    rec_icu, guard_icu = evaluated[0]
    assert rec_icu.target == "EQ-CRIT-ICU-01"
    assert guard_icu.status == "blocked"
    assert guard_icu.safe_alternative is not None

    rec_ahu, guard_ahu = evaluated[1]
    assert rec_ahu.target == "BLD-101"
    assert guard_ahu.status == "pass"


def test_action_dispatch_to_ledger(mock_facility_profile):
    """Test dispatching parsed actions to SavingsLedger."""
    ledger = SavingsLedger()
    actions = [
        {
            "action_type": "HVAC Inspection/Schedule",
            "target_asset": "BLD-101",
            "suggested_time": "Immediately",
            "estimated_saving_kwh": 40.0,
        }
    ]

    recs = actions_to_recommendations(actions, mock_facility_profile)
    rec = recs[0]

    # 1. Add to Decisions Queue (Admin flow)
    item_dec = ledger.add_recommendation(rec)
    assert item_dec.status == "proposed"
    assert item_dec.recommendation_id == rec.id

    # 2. Create Work Order (Operations flow)
    wo = ledger.create_work_order(item_dec.work_order_id, assignee="HVAC Specialist", sla_hours=24)
    assert wo.status == "work_order"
    assert wo.assignee == "HVAC Specialist"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Panel Rendering Graceful Execution
# ─────────────────────────────────────────────────────────────────────────────

def test_panel_render_force_open(mock_facility_profile, mock_telemetry_dfs):
    """Test that render_ask_nivara_panel executes gracefully when triggered."""
    assets, tel, events, complaints = mock_telemetry_dfs
    user = UserProfile(id="u1", name="Arun Patel", role="operations", email="arun@campus.ac.in", title="Operations Lead", facilities=["*"])
    ledger = SavingsLedger()

    # Should not raise exception
    render_ask_nivara_panel(
        current_user=user,
        facility_meta=mock_facility_profile,
        assets=assets,
        tel=tel,
        events=events,
        complaints=complaints,
        ledger=ledger,
        force_open=True,
    )
