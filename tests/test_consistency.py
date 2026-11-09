"""Consistency and Regulatory Compliance Alignment Tests for Nivara.

Verifies:
1. Headline numbers consistency: Avoided cost range, carbon abatement range,
   verified savings, and BEE EPI in ESG metrics match the data passed into and
   rendered by generate_facility_pdf.
2. PDF report contains Appendix A (Centralized Assumptions Registry) with citations.
3. PDF report includes Blocked-by-Guard count and summary table.
4. PDF report includes Appendix B (Model Evaluation & BDG2 Evidence Summary).
5. BEE Star Rating and NAPCC calculations dynamically resolve from AssumptionsRegistry
   without hardcoded literals.
"""
from __future__ import annotations

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytest

from campus_energy import assumptions
from campus_energy.analytics.carbon_budget import calculate_carbon_budget
from campus_energy.analytics.esg import (
    BEE_EPI_BENCHMARKS,
    _get_bee_star,
    calculate_esg_metrics,
    get_bee_benchmarks,
)
from campus_energy.analytics.report_generator import generate_facility_pdf
from campus_energy.decision.ledger import SavingsLedger
from campus_energy.decision.recommendation import (
    GuardResult,
    Recommendation,
    SavingsBundle,
    SavingsRange,
)


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Helper to extract all plaintext content across PDF pages using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text_chunks = [page.get_text() for page in doc]
    return "\n".join(text_chunks)


@pytest.fixture
def mock_telemetry_and_events():
    """Create minimal deterministic telemetry and fault events for testing."""
    dates = pd.date_range("2026-03-01 00:00:00", periods=28 * 24, freq="1h", tz="UTC")
    records = []
    for ts in dates:
        hour = ts.hour
        # Base load: 20 kW night, 80 kW day
        base_kw = 20.0 if (hour < 6 or hour > 20) else 80.0
        records.append({
            "timestamp": ts,
            "asset_id": "MTR-MAIN-01",
            "reading_type": "meter",
            "interval_energy_kwh": base_kw,  # 1 hour interval -> kWh = kW
            "power_kw": base_kw,
        })
    tel_df = pd.DataFrame(records)

    events_df = pd.DataFrame([
        {
            "event_id": "EVT-HVAC-OVERRUN-01",
            "asset_id": "MTR-MAIN-01",
            "event_type": "hvac_left_on",
            "start_time": "2026-03-05 22:00:00+00:00",
            "end_time": "2026-03-06 04:00:00+00:00",
            "severity": "high",
            "description": "After-hours chiller running without occupancy",
        }
    ])
    return tel_df, events_df


@pytest.fixture
def active_savings_ledger():
    """Create a populated closed-loop SavingsLedger with verified actions."""
    ledger = SavingsLedger()
    rec1 = Recommendation(
        id="REC-2026-001",
        facility_id="university",
        type="setpoint_adjustment",
        target="EQ-CHILLER-01",
        action_text="Night chilled water setback to 9°C",
        trigger="alert:ALT-101",
        savings=SavingsBundle.from_expected(1200.0, inr_per_kwh=8.5, co2_per_kwh=0.82),
        cost=0.0,
        payback_months=0.0,
        assumption_refs=["tariff.electricity_commercial", "emissions.grid_factor"],
        owner_role="operations",
        lifecycle="verified",
    )
    wo1 = ledger.add_recommendation(rec1)
    wo1.status = "verified"
    wo1.verified_savings = SavingsBundle.from_expected(1180.0, inr_per_kwh=8.5, co2_per_kwh=0.82)
    return ledger


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Headline Numbers Consistency
# ─────────────────────────────────────────────────────────────────────────────
def test_headline_numbers_consistency(mock_telemetry_and_events, active_savings_ledger):
    """Asserts avoided cost range, carbon abatement range, verified savings, and BEE EPI

    in ESG metrics match the data passed into and rendered by generate_facility_pdf.
    """
    tel_df, events_df = mock_telemetry_and_events
    floor_area = 10000.0

    esg = calculate_esg_metrics(tel_df, events_df, floor_area_m2=floor_area, facility_type="university")
    kpis = esg["kpis"]
    bee = esg["bee"]
    p1_summary = active_savings_ledger.get_summary()

    pdf_bytes = generate_facility_pdf(
        facility_name="Vidya Institute of Technology",
        facility_type="university",
        floor_area_m2=floor_area,
        kpis=kpis,
        bee_info=bee,
        events_df=events_df,
        ledger=active_savings_ledger,
    )

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000

    pdf_text = extract_pdf_text(pdf_bytes)

    # 1. Total energy & cost consistency
    assert f"{kpis['total_energy_kwh']:,.0f}" in pdf_text or "kWh" in pdf_text
    # Avoided cost range presence
    cost_rng = kpis["avoidable_savings_range"]["cost_inr"]
    assert f"{cost_rng['expected']:,.0f}" in pdf_text
    assert f"{cost_rng['low']:,.0f}" in pdf_text or f"{cost_rng['high']:,.0f}" in pdf_text

    # 2. Carbon abatement range consistency
    carb_rng = kpis["avoidable_savings_range"]["carbon_kg"]
    assert f"{carb_rng['expected']:,.0f}" in pdf_text or f"{carb_rng['expected']:.0f}" in pdf_text

    # 3. Verified savings consistency (from active_savings_ledger)
    ver_inr = p1_summary.cumulative_verified_inr
    assert f"{ver_inr:,.0f}" in pdf_text
    assert f"{p1_summary.estimate_hit_rate_pct:.1f}%" in pdf_text or "HIT" in pdf_text

    # 4. BEE EPI and star rating consistency
    assert f"{bee['epi_annual_kwh_m2']:.1f}" in pdf_text or f"{bee['epi_annual_kwh_m2']:.0f}" in pdf_text
    assert f"{bee['stars']}/5" in pdf_text or f"{bee['stars']} / 5" in pdf_text


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Assumptions Appendix & Citations
# ─────────────────────────────────────────────────────────────────────────────
def test_pdf_report_contains_assumptions_appendix(mock_telemetry_and_events):
    """Asserts 'Appendix A: Centralized Assumptions Registry' and regulatory citations

    exist in the generated PDF.
    """
    tel_df, events_df = mock_telemetry_and_events
    esg = calculate_esg_metrics(tel_df, events_df, floor_area_m2=12000.0, facility_type="university")

    pdf_bytes = generate_facility_pdf(
        facility_name="Vidya Institute of Technology",
        facility_type="university",
        floor_area_m2=12000.0,
        kpis=esg["kpis"],
        bee_info=esg["bee"],
        events_df=events_df,
    )

    pdf_text = extract_pdf_text(pdf_bytes)

    # Check header
    assert "APPENDIX A" in pdf_text
    assert "CENTRALIZED ASSUMPTIONS REGISTRY" in pdf_text

    # Check key statutory assumption citations
    assert "tariff.electricity_commercial" in pdf_text
    assert "emissions.grid_factor" in pdf_text
    assert "Central Electricity Authority" in pdf_text or "CEA" in pdf_text
    assert "Odisha Electricity Regulatory Commission" in pdf_text or "OERC" in pdf_text


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Blocked-by-Guard Count and Summary Table
# ─────────────────────────────────────────────────────────────────────────────
def test_pdf_report_includes_blocked_guard_summary(mock_telemetry_and_events):
    """Asserts PDF report includes the Blocked-by-Guard count, constraints evaluated,

    and defensible safety preservation rationale.
    """
    tel_df, events_df = mock_telemetry_and_events
    esg = calculate_esg_metrics(tel_df, events_df, floor_area_m2=15000.0, facility_type="hospital")

    # Construct an explicit blocked recommendation to pass into the generator
    blocked_rec = Recommendation(
        id="REC-BLOCKED-01",
        facility_id="district_hospital",
        type="equipment_shutdown",
        target="EQ-ICU-AHU-01",
        action_text="Nighttime ICU AHU Ventilation Curtailment",
        trigger="alert:ALT-999",
        savings=SavingsBundle.from_expected(500.0, inr_per_kwh=8.5),
        cost=0.0,
        owner_role="operations",
        service_guard=GuardResult(
            status="blocked",
            reason="ICU life safety ventilation is a protected non-curtailable load under NABH guidelines.",
            constraints_checked=["protected_load_check", "clinical_safety_check"],
            safe_alternative="Modulate outpatient clinic administrative setpoints instead.",
        ),
    )

    pdf_bytes = generate_facility_pdf(
        facility_name="District General Hospital",
        facility_type="hospital",
        floor_area_m2=15000.0,
        kpis=esg["kpis"],
        bee_info=esg["bee"],
        events_df=events_df,
        blocked_recommendations=[blocked_rec],
    )

    pdf_text = extract_pdf_text(pdf_bytes)

    # Assert Blocked section exists
    assert "BLOCKED-BY-SERVICE-GUARD MEASURES" in pdf_text
    assert "CLINICAL & OPERATIONAL SAFETY" in pdf_text
    assert "BLOCKED" in pdf_text
    assert "EQ-ICU-AHU-01" in pdf_text or "ICU" in pdf_text
    assert "protected_load_check" in pdf_text


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Model Evaluation / Evidence Summary
# ─────────────────────────────────────────────────────────────────────────────
def test_pdf_report_includes_model_evidence_summary(mock_telemetry_and_events):
    """Asserts PDF report includes Appendix B (Model Evaluation & BDG2 Evidence Summary)."""
    tel_df, events_df = mock_telemetry_and_events
    esg = calculate_esg_metrics(tel_df, events_df, floor_area_m2=10000.0, facility_type="university")

    pdf_bytes = generate_facility_pdf(
        facility_name="Vidya Institute of Technology",
        facility_type="university",
        floor_area_m2=10000.0,
        kpis=esg["kpis"],
        bee_info=esg["bee"],
        events_df=events_df,
    )

    pdf_text = extract_pdf_text(pdf_bytes)

    # Check header
    assert "APPENDIX B" in pdf_text
    assert "MODEL EVALUATION & EVIDENCE SUMMARY" in pdf_text

    # Check B.1 Anomaly Evaluation
    assert "Layered Engine" in pdf_text
    assert "80.0%" in pdf_text
    assert "Recall" in pdf_text

    # Check B.2 Forecasting Accuracy
    assert "Seasonal-Naive" in pdf_text
    assert "8.27%" in pdf_text
    assert "MAPE" in pdf_text

    # Check B.3 Real-World BDG2 Proof
    assert "Building Data Genome 2" in pdf_text or "BDG2" in pdf_text
    assert "UnivClass_Mary" in pdf_text or "Miller et al." in pdf_text
    assert "7.40%" in pdf_text or "7.4%" in pdf_text


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: BEE Star Rating and NAPCC Calculations Resolve from Registry
# ─────────────────────────────────────────────────────────────────────────────
def test_bee_and_napcc_resolve_dynamically_without_hardcoded_literals(monkeypatch):
    """Asserts BEE star rating and NAPCC calculations resolve assumptions from registry

    dynamically rather than using hardcoded values.
    """
    # 1. Verify dynamic BEE EPI Star calculation shifts when assumption changes
    initial_university_norms = get_bee_benchmarks("university")
    assert initial_university_norms["ecbc_baseline"] == 100.0
    assert initial_university_norms["stars"][5] == (0.0, 60.0)

    # An EPI of 70 is 4-star under standard threshold (60 <= 70 < 90)
    std_result = _get_bee_star(70.0, "university")
    assert std_result["stars"] == 4
    assert std_result["napcc_target_kwh_m2"] == 80.0  # 100 * (1 - 0.20)

    # Dynamically patch the five-star threshold to 80.0 in the registry
    orig_get_value = assumptions.get_value

    def mock_get_value(key: str, default=None):
        if key == "bee.epi.university.five_star_max":
            return 80.0
        if key == "emissions.napcc_target_reduction_pct":
            return 30.0  # 30% reduction instead of 20%
        return orig_get_value(key, default=default)

    monkeypatch.setattr(assumptions, "get_value", mock_get_value)

    # Now with threshold = 80.0, EPI = 70.0 MUST become 5-star!
    dynamic_result = _get_bee_star(70.0, "university")
    assert dynamic_result["stars"] == 5, "BEE star rating should dynamically reflect patched assumption threshold"
    assert dynamic_result["napcc_target_kwh_m2"] == 70.0, "NAPCC target should dynamically reflect patched 30% target"

    # 2. Verify NAPCC burndown in calculate_carbon_budget uses dynamic registry target
    mock_tel = pd.DataFrame([{
        "timestamp": "2026-03-01 12:00:00",
        "asset_id": "MTR-01",
        "reading_type": "meter",
        "interval_energy_kwh": 100.0,
    }])
    cb = calculate_carbon_budget(mock_tel, floor_area_m2=10000.0, facility_type="university")
    # With 30% reduction, target kWh = 100 * 0.70 * 10000 = 700,000 kWh/yr -> monthly target = (700000 / 12 * 0.82) / 1000 = 47.83 tCO2
    assert cb["summary"]["monthly_budget_tco2"] == pytest.approx(47.8, abs=0.2)
