"""Unit tests for the Executive & Regulatory PDF Report Generator."""
from pathlib import Path
import pandas as pd
import pytest

from campus_energy.analytics.report_generator import generate_facility_pdf


@pytest.fixture
def mock_kpis():
    return {
        "total_energy_kwh": 52400.0,
        "total_cost_inr": 445400.0,
        "total_carbon_kg": 42968.0,
        "total_carbon_tons": 42.97,
        "wasted_kwh": 3200.0,
        "avoidable_cost_inr": 27200.0,
        "avoidable_carbon_kg": 2624.0,
        "waste_pct": 6.1,
        "annual_kwh_projection": 682000.0,
        "annual_cost_inr_projection": 5797000.0,
        "annual_carbon_tons_projection": 559.24,
        "trees_equivalent": 26630,
        "napcc_savings_kwh": 136400.0,
        "napcc_savings_cost_inr": 1159400.0,
        "napcc_savings_carbon_tons": 111.85,
    }


@pytest.fixture
def mock_bee_compliant():
    return {
        "stars": 4,
        "epi_annual_kwh_m2": 56.8,
        "ecbc_baseline_kwh_m2": 100.0,
        "vs_ecbc_pct": -43.2,
        "napcc_target_kwh_m2": 80.0,
        "facility_label": "Educational Building",
        "compliant": True,
    }


@pytest.fixture
def mock_bee_non_compliant():
    return {
        "stars": 1,
        "epi_annual_kwh_m2": 175.5,
        "ecbc_baseline_kwh_m2": 100.0,
        "vs_ecbc_pct": 75.5,
        "napcc_target_kwh_m2": 80.0,
        "facility_label": "Educational Building",
        "compliant": False,
    }


@pytest.fixture
def mock_events_df():
    return pd.DataFrame([
        {
            "event_id": "EVT-EED9E6D3",
            "event_type": "lights_left_on",
            "asset_id": "EQ-A-CR101-LT",
            "start_time": "2025-09-03T18:00:00+05:30",
            "end_time": "2025-09-04T00:00:00+05:30",
            "severity": "medium",
            "description": "Lights left on overnight in Classroom 101",
        },
        {
            "event_id": "EVT-54D71E1B",
            "event_type": "hvac_left_on",
            "asset_id": "EQ-ADM-OF101-AC",
            "start_time": "2025-09-06T19:00:00+05:30",
            "end_time": "2025-09-07T03:00:00+05:30",
            "severity": "high",
            "description": "Office HVAC running after hours",
        },
        {
            "event_id": "EVT-99C12A34",
            "event_type": "sub_meter_drift",
            "asset_id": "MTR-BLD-B",
            "start_time": "2025-09-10T10:00:00+05:30",
            "end_time": "2025-09-10T18:00:00+05:30",
            "severity": "critical",
            "description": "Sub-meter baseline drift detected",
        },
    ])


def test_generate_facility_pdf_returns_valid_pdf(mock_kpis, mock_bee_compliant, mock_events_df):
    """Test that generate_facility_pdf returns valid PDF bytes (> 1000 bytes and starting with %PDF)."""
    pdf_bytes = generate_facility_pdf(
        facility_name="🎓 University — Vidya Institute of Technology",
        facility_type="university",
        floor_area_m2=12000.0,
        kpis=mock_kpis,
        bee_info=mock_bee_compliant,
        events_df=mock_events_df,
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_generate_facility_pdf_with_none_events(mock_kpis, mock_bee_compliant):
    """Test PDF generation when events_df is None."""
    pdf_bytes = generate_facility_pdf(
        facility_name="Vidya Institute of Technology",
        facility_type="university",
        floor_area_m2=12000.0,
        kpis=mock_kpis,
        bee_info=mock_bee_compliant,
        events_df=None,
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_generate_facility_pdf_with_empty_events(mock_kpis, mock_bee_compliant):
    """Test PDF generation when events_df is an empty DataFrame."""
    pdf_bytes = generate_facility_pdf(
        facility_name="Vidya Institute of Technology",
        facility_type="university",
        floor_area_m2=12000.0,
        kpis=mock_kpis,
        bee_info=mock_bee_compliant,
        events_df=pd.DataFrame(),
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_generate_facility_pdf_non_compliant(mock_kpis, mock_bee_non_compliant):
    """Test PDF generation with a non-compliant 1-star facility."""
    pdf_bytes = generate_facility_pdf(
        facility_name="Non-Compliant Complex",
        facility_type="govt_office",
        floor_area_m2=5200.0,
        kpis=mock_kpis,
        bee_info=mock_bee_non_compliant,
        events_df=None,
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


@pytest.mark.parametrize("fac_type,fac_label,area", [
    ("university", "Educational Building", 12000.0),
    ("hospital", "Healthcare Facility", 8500.0),
    ("govt_office", "Government Office", 5200.0),
    ("industrial", "Industrial Facility", 18000.0),
])
def test_generate_facility_pdf_all_facility_types(mock_kpis, fac_type, fac_label, area):
    """Test PDF generation across all standard facility types."""
    bee_info = {
        "stars": 3,
        "epi_annual_kwh_m2": 110.0,
        "ecbc_baseline_kwh_m2": 120.0,
        "vs_ecbc_pct": -8.3,
        "napcc_target_kwh_m2": 96.0,
        "facility_label": fac_label,
        "compliant": True,
    }
    pdf_bytes = generate_facility_pdf(
        facility_name=f"Sample {fac_label}",
        facility_type=fac_type,
        floor_area_m2=area,
        kpis=mock_kpis,
        bee_info=bee_info,
        events_df=None,
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_generate_facility_pdf_real_data_integration():
    """Test integration using actual simulation data if available."""
    data_dir = Path("data/generated")
    tel_path = data_dir / "telemetry.csv"
    events_path = data_dir / "events.csv"

    if not tel_path.exists() or not events_path.exists():
        pytest.skip("Generated dataset not present on disk")

    from campus_energy.analytics.esg import calculate_esg_metrics

    tel = pd.read_csv(tel_path)
    events = pd.read_csv(events_path)
    esg = calculate_esg_metrics(tel, events, floor_area_m2=12000.0, facility_type="university")

    pdf_bytes = generate_facility_pdf(
        facility_name="Vidya Institute of Technology",
        facility_type="university",
        floor_area_m2=12000.0,
        kpis=esg["kpis"],
        bee_info=esg["bee"],
        events_df=events.head(5),
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 2000


def test_generate_facility_pdf_with_savings_ledger(mock_kpis, mock_bee_compliant):
    """Test PDF generation when a SavingsLedger with verified items is provided."""
    from campus_energy.decision.ledger import SavingsLedger
    from campus_energy.decision.recommendation import Recommendation, SavingsBundle

    rec = Recommendation(
        id="REC-2026-PDF",
        facility_id="university",
        type="lighting_shutdown",
        target="EQ-LT-101",
        action_text="Night setback lighting schedule",
        trigger="alert:ALT-101",
        savings=SavingsBundle.from_expected(120.0, inr_per_kwh=8.50),
        cost=0.0,
        payback_months=0.0,
        assumption_refs=["tariff.electricity_commercial"],
        owner_role="operations",
        lifecycle="verified",
        confidence=0.9,
    )
    ledger = SavingsLedger()
    wo = ledger.add_recommendation(rec)
    wo.status = "verified"
    wo.verified_savings = SavingsBundle.from_expected(115.0, inr_per_kwh=8.50)

    pdf_bytes = generate_facility_pdf(
        facility_name="Vidya Institute of Technology",
        facility_type="university",
        floor_area_m2=12000.0,
        kpis=mock_kpis,
        bee_info=mock_bee_compliant,
        ledger=ledger,
    )

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 2000

