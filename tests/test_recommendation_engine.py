"""Unit and Property Tests for Unified Recommendation Engine & Savings Model (Step 8)."""
from __future__ import annotations

import pandas as pd
import pytest

from campus_energy import assumptions
from campus_energy.dashboard.agent import (
    actions_to_recommendations,
    generate_guarded_agent_recommendations,
    parse_actions_from_response,
)
from campus_energy.decision import (
    Recommendation,
    RecommendationEngine,
    SavingsBundle,
    apply_service_guard,
    build_ranked_recommendations,
    calculate_savings,
    check_service_guard,
    generate_recommendations_from_alerts,
    generate_recommendations_from_complaints,
    generate_recommendations_from_forecast,
    generate_recommendations_from_scenarios,
    get_benchmark_capital_cost,
    get_savings_assumption_keys,
)
from campus_energy.profiles import load_facility_profile
from campus_energy.schemas import Alert, AlertStatus, Severity


@pytest.fixture
def hospital_profile():
    return load_facility_profile("config/facility_profiles/district_hospital.yaml")


@pytest.fixture
def university_profile():
    return load_facility_profile("config/facility_profiles/university.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Savings & ROI Model Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_savings_calculation_ranges_monotonicity():
    """Property test: For any expected kWh, low <= expected <= high for kWh, INR, and CO2."""
    test_cases = [0.0, 5.0, 25.5, 100.0, 1500.0, 50000.0]
    for kwh in test_cases:
        bundle, payback = calculate_savings(expected_kwh=kwh, variance_pct=0.20)

        # kWh monotonicity
        assert bundle.kwh.low <= bundle.kwh.expected <= bundle.kwh.high
        assert bundle.kwh.low >= 0.0

        # INR monotonicity
        assert bundle.inr.low <= bundle.inr.expected <= bundle.inr.high
        assert bundle.inr.low >= 0.0

        # CO2 monotonicity
        assert bundle.co2_kg.low <= bundle.co2_kg.expected <= bundle.co2_kg.high
        assert bundle.co2_kg.low >= 0.0

        # Payback should be None when capital cost is zero
        assert payback is None


def test_savings_with_water():
    """Verify water savings convert to INR using assumptions registry municipal water tariff."""
    bundle, _ = calculate_savings(expected_kwh=100.0, water_kl=10.0, variance_pct=0.20)
    tariff_elec = float(assumptions.get_value("tariff.electricity_commercial"))
    tariff_water = float(assumptions.get_value("tariff.water_municipal"))

    expected_inr = round(100.0 * tariff_elec + 10.0 * tariff_water, 1)
    assert bundle.inr.expected == expected_inr
    assert bundle.inr.low <= bundle.inr.expected <= bundle.inr.high


def test_payback_calculation_monotonicity():
    """Test payback formula and monotonicity properties:
    1. Capital cost > 0 yields positive payback period.
    2. Higher capital cost yields strictly longer payback.
    3. Higher annual savings yields strictly shorter payback.
    """
    # Payback > 0 for positive capex
    b1, pb1 = calculate_savings(expected_kwh=1000.0, capital_cost=50000.0, days_modeled=365.0)
    assert pb1 is not None
    assert pb1 > 0.0

    # Payback increases with higher capital cost
    b2, pb2 = calculate_savings(expected_kwh=1000.0, capital_cost=100000.0, days_modeled=365.0)
    assert pb2 is not None
    assert pb2 > pb1

    # Payback decreases with higher savings (same capital cost)
    b3, pb3 = calculate_savings(expected_kwh=2000.0, capital_cost=50000.0, days_modeled=365.0)
    assert pb3 is not None
    assert pb3 < pb1


def test_capital_cost_benchmarks_resolve_in_assumptions():
    """Verify all benchmark capital costs resolve accurately from config/assumptions.yaml."""
    benchmarks = ["led_retrofit", "solar_pv", "bess", "hvac_vfd", "smart_sensor"]
    for b in benchmarks:
        cost = get_benchmark_capital_cost(b, quantity=10.0)
        assert cost > 0.0

    keys = get_savings_assumption_keys(include_water=True, capex_measure="led_retrofit")
    assert "tariff.electricity_commercial" in keys
    assert "emissions.grid_factor" in keys
    assert "tariff.water_municipal" in keys
    assert "capital_cost.led_retrofit_per_kw" in keys
    for k in keys:
        assert assumptions.get(k) is not None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Recommendations from Multi-Source Streams
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_recommendations_from_alerts(university_profile):
    """Test generating typed recommendations from diverse anomaly alerts."""
    alerts = [
        {
            "alert_id": "ALT-01",
            "anomaly_type": "unoccupied_load",
            "asset_id": "BLD-ACAD-LTG",
            "recommendation": "Turn off corridor lights in Academic Block during off-hours",
            "deviation": 8.0,
        },
        {
            "alert_id": "ALT-02",
            "anomaly_type": "unoccupied_load",
            "asset_id": "BLD-ACAD-HVAC",
            "recommendation": "Apply night setback to Academic Block central AC",
            "deviation": 25.0,
        },
        {
            "alert_id": "ALT-03",
            "anomaly_type": "pump_dry_run",
            "asset_id": "PMP-MAIN-01",
            "recommendation": "Inspect pump for cavitation and dry-run",
            "deviation": 12.0,
        },
        {
            "alert_id": "ALT-04",
            "anomaly_type": "sensor_dropout",
            "asset_id": "MTR-SUB-102",
            "recommendation": "Check gateway telemetry sensor power",
            "deviation": 0.0,
        },
        {
            "alert_id": "ALT-05",
            "anomaly_type": "water_leak",
            "asset_id": "WTR-MTR-02",
            "recommendation": "Isolate pipeline rupture and repair",
            "deviation": 15.0,
        },
    ]

    recs = generate_recommendations_from_alerts(alerts, profile=university_profile)
    assert len(recs) == 5

    types = [r.type for r in recs]
    assert "lighting_shutdown" in types
    assert "hvac_shutdown" in types
    assert "equipment_shutdown" in types
    assert "sensor_check" in types
    assert "water_leak_repair" in types

    for r in recs:
        assert r.owner_role == "operations"
        assert r.is_estimate is True
        assert len(r.assumption_refs) >= 2
        # Check range monotonicity property
        assert r.savings.kwh.low <= r.savings.kwh.expected <= r.savings.kwh.high
        assert r.savings.inr.low <= r.savings.inr.expected <= r.savings.inr.high
        assert r.savings.co2_kg.low <= r.savings.co2_kg.expected <= r.savings.co2_kg.high


def test_generate_recommendations_from_forecast(university_profile):
    """Test detecting peak demand surges in predictive forecast and synthesizing peak shaving actions."""
    hours = list(range(24))
    powers = [40.0 + (120.0 if 17 <= h <= 20 else 10.0) for h in hours]
    df = pd.DataFrame({
        "hour": hours,
        "predicted_power_kw": powers,
        "asset_id": ["Main Substation"] * 24,
    })

    recs = generate_recommendations_from_forecast(df, profile=university_profile)
    assert len(recs) >= 1
    rec = recs[0]
    assert rec.type == "peak_shaving"
    assert "BESS" in rec.action_text or "peak" in rec.action_text.lower()
    assert rec.savings.inr.expected > 0.0
    assert rec.owner_role in ("operations", "administrator")
    assert "tariff.tod_peak" in rec.assumption_refs


def test_generate_recommendations_from_scenarios(university_profile):
    """Test generating capital/policy recommendations from scenarios."""
    # 1. LED upgrade
    recs_led = generate_recommendations_from_scenarios(
        "led_upgrade", {"led_upgrade": True, "lighting_kw": 80.0}, profile=university_profile
    )
    assert len(recs_led) == 1
    r_led = recs_led[0]
    assert r_led.type == "led_upgrade"
    assert r_led.cost > 0.0
    assert r_led.payback_months is not None and r_led.payback_months > 0.0
    assert r_led.owner_role == "sustainability"
    assert "capital_cost.led_retrofit_per_kw" in r_led.assumption_refs

    # 2. Solar PV
    recs_solar = generate_recommendations_from_scenarios(
        "solar_pv", {"solar_pv_kwp": 200.0}, profile=university_profile
    )
    assert len(recs_solar) == 1
    r_solar = recs_solar[0]
    assert r_solar.type == "solar_bess"
    assert r_solar.cost > 0.0
    assert r_solar.payback_months is not None and r_solar.payback_months > 0.0
    assert "capital_cost.solar_pv_per_kwp" in r_solar.assumption_refs

    # 3. Thermostat setpoint adjustment (0 capex)
    recs_temp = generate_recommendations_from_scenarios(
        "setpoint_adjustment", {"temp_offset": 1.5}, profile=university_profile
    )
    assert len(recs_temp) == 1
    r_temp = recs_temp[0]
    assert r_temp.type == "setpoint_adjustment"
    assert r_temp.cost == 0.0
    assert r_temp.payback_months is None


def test_generate_recommendations_from_complaints(university_profile):
    """Test generating operational maintenance recommendations from complaint clusters."""
    clusters = [
        {"theme_label": "Thermal Comfort Stuffy", "count": 14, "target": "Academic Wing C"},
        {"theme_label": "Lighting Flickering", "count": 6, "target": "Library Hall"},
        {"theme_label": "Water Pipe Joint Leak", "count": 4, "target": "Hostel Riser B"},
    ]
    recs = generate_recommendations_from_complaints(clusters, profile=university_profile)
    assert len(recs) == 3

    types = [r.type for r in recs]
    assert "setpoint_adjustment" in types
    assert "lighting_audit" in types
    assert "water_leak_repair" in types

    for r in recs:
        assert r.owner_role == "operations"
        assert r.savings.inr.expected > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Master Aggregator, Guard Isolation Property, and Engine Class
# ─────────────────────────────────────────────────────────────────────────────

def test_build_ranked_recommendations_isolation_property(hospital_profile):
    """Crucial Property Test:
    1. Guard blocks life-safety / clinical protected load actions.
    2. Blocked recommendations are completely isolated into blocked_recommendations.
    3. Blocked recommendations NEVER appear in the approved ranked action list.
    4. Approved recommendations are strictly sorted by expected INR savings descending.
    """
    # Create alerts containing both safe actions and forbidden clinical actions
    mixed_alerts = [
        # Safe action: Administration corridor lighting
        {
            "alert_id": "ALT-SAFE-01",
            "anomaly_type": "unoccupied_load",
            "asset_id": "EQ-ADM-LTG",
            "recommendation": "Turn off corridor lights in Administration Block after 19:00",
            "deviation": 12.0,
        },
        # Safe action: Staff cafeteria AC setback
        {
            "alert_id": "ALT-SAFE-02",
            "anomaly_type": "unoccupied_load",
            "asset_id": "EQ-CAF-AC",
            "recommendation": "Apply night setback to Staff Cafeteria AC",
            "deviation": 30.0,
        },
        # FORBIDDEN: ICU Life Support Curtailment
        {
            "alert_id": "ALT-BLOCKED-01",
            "anomaly_type": "unoccupied_load",
            "asset_id": "EQ-IPD-ICU-EQ",
            "recommendation": "Shut down or curtail power to ICU life support equipment during peak tariff",
            "deviation": 50.0,
        },
        # FORBIDDEN: Cold-Chain Vaccine Refrigerator
        {
            "alert_id": "ALT-BLOCKED-02",
            "anomaly_type": "unoccupied_load",
            "asset_id": "EQ-IPD-CCR-AC",
            "recommendation": "Shut down cold chain vaccine refrigerator for 2 hours",
            "deviation": 40.0,
        },
    ]

    engine = RecommendationEngine(hospital_profile)
    approved, blocked = engine.build_ranked(alerts=mixed_alerts)

    assert len(approved) >= 2
    assert len(blocked) >= 2

    # Verify blocked actions are indeed blocked
    blocked_ids = {r.id for r in blocked}
    blocked_targets = {r.target for r in blocked}
    assert "EQ-IPD-ICU-EQ" in blocked_targets
    assert "EQ-IPD-CCR-AC" in blocked_targets

    # STRICT ISOLATION PROPERTY: No blocked action may be in approved
    for app_rec in approved:
        assert app_rec.id not in blocked_ids
        assert app_rec.target not in {"EQ-IPD-ICU-EQ", "EQ-IPD-CCR-AC"}
        assert not app_rec.is_blocked
        assert app_rec.service_guard is not None
        assert app_rec.service_guard.status == "pass"

    # STRICT RANKING PROPERTY: Approved actions sorted by expected INR savings descending
    for i in range(len(approved) - 1):
        assert approved[i].savings.inr.expected >= approved[i + 1].savings.inr.expected


def test_all_recommendations_have_traceable_assumptions(university_profile):
    """Verify that every recommendation generated by engine has assumption_refs that exist in assumptions registry."""
    engine = RecommendationEngine(university_profile)
    alerts = [
        {"alert_id": "A1", "anomaly_type": "unoccupied_load", "asset_id": "BLD-1", "deviation": 10.0},
        {"alert_id": "A2", "anomaly_type": "water_leak", "asset_id": "WTR-1", "deviation": 5.0},
    ]
    scenarios = [
        ("led_upgrade", {"led_upgrade": True}),
        ("solar_pv", {"solar_pv_kwp": 50.0}),
    ]

    approved, blocked = engine.build_ranked(alerts=alerts, scenarios=scenarios)
    all_recs = approved + blocked

    assert len(all_recs) >= 4
    for r in all_recs:
        assert r.is_estimate is True
        assert len(r.assumption_refs) > 0
        for ref_key in r.assumption_refs:
            # Must not raise KeyError
            item = assumptions.get(ref_key)
            assert item is not None
            assert item.key == ref_key


def test_agent_actions_to_recommendations_and_guard(hospital_profile):
    """Test translating agent actions to validated recommendations and applying service guard."""
    agent_text = (
        "You should urgently shutdown the ICU equipment in EQ-IPD-ICU-EQ immediately to save 50 kWh. "
        "Also audit the lighting in Administration Block after hours."
    )
    raw_actions = parse_actions_from_response(agent_text)
    assert len(raw_actions) >= 1

    approved, blocked = generate_guarded_agent_recommendations(raw_actions, hospital_profile)

    # Any attempt to shutdown ICU must be blocked
    icu_blocked = [r for r in blocked if "ICU" in r.action_text or "EQ-IPD-ICU-EQ" in r.target]
    if any("ICU" in a.get("action_type", "") or "EQ-IPD-ICU-EQ" in a.get("target_asset", "") for a in raw_actions):
        assert len(icu_blocked) > 0
        assert icu_blocked[0].service_guard.status == "blocked"

    # Verify that approved recommendations have valid, non-invented savings
    for r in approved:
        assert r.savings.inr.expected > 0.0
        assert r.service_guard.status == "pass"
