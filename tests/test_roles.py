"""Unit tests for Nivara Role-Based Views & Personas (Step 12)."""
import pytest
from pathlib import Path

from campus_energy.roles import (
    RoleDefinition,
    RolesRegistry,
    load_roles,
    get_role,
    VALID_FOCUS_WIDGETS,
)
from campus_energy.dashboard.agent import build_system_prompt


def test_load_roles_all_primary_personas():
    """Validate that config/roles.yaml loads all 3 primary operational roles."""
    roles = load_roles("config/roles.yaml")
    assert isinstance(roles, dict)
    assert set(roles.keys()) >= {"administrator", "operations", "sustainability"}

    # Administrator
    admin = roles["administrator"]
    assert admin.name == "Facility Administrator & Registrar"
    assert admin.icon == "👔"
    assert admin.default_page == "📊  Executive Overview"
    assert "executive_kpis" in admin.focus_widgets
    assert "tariff_summary" in admin.focus_widgets
    assert "ranked_actions" in admin.focus_widgets
    assert "ledger_totals" in admin.focus_widgets
    assert "compliance_scorecard" in admin.focus_widgets
    assert "department_billing" in admin.focus_widgets
    assert "financial ROI" in admin.gemini_persona_prompt

    # Operations
    ops = roles["operations"]
    assert ops.name == "Operations & Maintenance Lead"
    assert ops.icon == "🔧"
    assert ops.default_page == "🚨  Alerts & NLP"
    assert "live_telemetry_feed" in ops.focus_widgets
    assert "active_anomalies" in ops.focus_widgets
    assert "why_diagnostic_cards" in ops.focus_widgets
    assert "work_order_queue" in ops.focus_widgets
    assert "sla_timers" in ops.focus_widgets
    assert "service_guard_blocks" in ops.focus_widgets
    assert "data_quality" in ops.focus_widgets
    assert "diagnostics specialist" in ops.gemini_persona_prompt

    # Sustainability
    sust = roles["sustainability"]
    assert sust.name == "Chief Sustainability Officer (CSO)"
    assert sust.icon == "🌱"
    assert sust.default_page == "🏛️  Governance & Policy"
    assert "esg_scorecard" in sust.focus_widgets
    assert "bee_star_rating" in sust.focus_widgets
    assert "napcc_carbon_burndown" in sust.focus_widgets
    assert "verified_impact_ledger" in sust.focus_widgets
    assert "microgrid_solar_bess" in sust.focus_widgets
    assert "verification_evidence" in sust.focus_widgets
    assert "Scope 2" in sust.gemini_persona_prompt


def test_roles_reference_only_valid_focus_widgets():
    """Validate that every role configuration references only documented valid focus widgets."""
    roles = load_roles("config/roles.yaml")
    for rk, role in roles.items():
        assert len(role.focus_widgets) > 0, f"Role {rk} has empty focus_widgets"
        for fw in role.focus_widgets:
            assert fw in VALID_FOCUS_WIDGETS, f"Widget '{fw}' in role '{rk}' is not in VALID_FOCUS_WIDGETS"


def test_get_role_valid_and_unknown():
    """Validate get_role returns correct model and raises KeyError on unknown key."""
    admin = get_role("administrator")
    assert isinstance(admin, RoleDefinition)
    assert admin.name == "Facility Administrator & Registrar"

    ops = get_role("operations")
    assert isinstance(ops, RoleDefinition)
    assert ops.icon == "🔧"

    sust = get_role("sustainability")
    assert isinstance(sust, RoleDefinition)
    assert sust.name == "Chief Sustainability Officer (CSO)"

    with pytest.raises(KeyError):
        get_role("unknown_role_xyz")


def test_roles_registry_model():
    """Validate RolesRegistry Pydantic model methods."""
    roles = load_roles("config/roles.yaml")
    reg = RolesRegistry(roles=roles)
    assert "administrator" in reg
    assert reg["operations"].icon == "🔧"
    assert reg.get("sustainability") is not None
    assert reg.get("nonexistent", "fallback") == "fallback"
    assert len(list(reg.keys())) >= 3


def test_build_system_prompt_injects_persona_prompts():
    """Validate build_system_prompt injects persona-specific prompts into Gemini system prompt."""
    prompt_admin = build_system_prompt("administrator")
    assert "Facility Administrator & Registrar" in prompt_admin
    assert "executive facility advisor to the Campus Administrator" in prompt_admin
    assert "financial ROI" in prompt_admin

    prompt_ops = build_system_prompt("operations")
    assert "Operations & Maintenance Lead" in prompt_ops
    assert "senior facility operations engineer and diagnostics specialist" in prompt_ops
    assert "SLA countdowns" in prompt_ops

    prompt_sust = build_system_prompt("sustainability")
    assert "Chief Sustainability Officer (CSO)" in prompt_sust
    assert "Chief Sustainability and ESG Officer" in prompt_sust
    assert "Scope 2" in prompt_sust


def test_build_system_prompt_default_is_administrator():
    """Validate build_system_prompt defaults to administrator when no role is passed."""
    prompt_default = build_system_prompt()
    assert "Facility Administrator & Registrar" in prompt_default
    assert "executive facility advisor to the Campus Administrator" in prompt_default
