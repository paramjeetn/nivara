"""Unit tests for the Governance & Policy module."""
from pathlib import Path
import pandas as pd
import pytest

from campus_energy.analytics.governance import (
    get_multi_facility_benchmarks,
    calculate_state_impact,
    clean_facility_label,
)

SAMPLE_FACILITY_OPTIONS = {
    "🎓 University — Vidya Institute of Technology": {
        "key": "university",
        "data_dir": Path("data/generated"),
        "floor_area_m2": 12000,
        "peak_occupancy": 2500,
    },
    "🏥 District Hospital — Rajiv Gandhi Hospital": {
        "key": "hospital",
        "data_dir": Path("data/hospital"),
        "floor_area_m2": 8500,
        "peak_occupancy": 420,
    },
    "🏛️ Government Office — Collectorate Complex": {
        "key": "govt_office",
        "data_dir": Path("data/govt_office"),
        "floor_area_m2": 5200,
        "peak_occupancy": 350,
    },
    "🏭 Industrial Campus — Kalinga Industrial Estate": {
        "key": "industrial",
        "data_dir": Path("data/industrial"),
        "floor_area_m2": 18000,
        "peak_occupancy": 600,
    },
}


def test_clean_facility_label():
    assert clean_facility_label("🎓 University — Vidya Institute of Technology") == "University"
    assert clean_facility_label("🏥 District Hospital — Rajiv Gandhi Hospital") == "District Hospital"
    assert clean_facility_label("🏛️ Government Office — Collectorate Complex") == "Government Office"
    assert clean_facility_label("🏭 Industrial Campus — Kalinga Industrial Estate") == "Industrial Campus"
    assert clean_facility_label("Custom Facility") == "Custom Facility"


def test_get_multi_facility_benchmarks():
    df = get_multi_facility_benchmarks(SAMPLE_FACILITY_OPTIONS)

    expected_columns = [
        "Facility",
        "Facility Type",
        "Floor Area (m²)",
        "Peak Occupancy",
        "Annual Energy (MWh)",
        "Annual Cost (₹ Lakh)",
        "EPI (kWh/m²/yr)",
        "ECBC Baseline",
        "BEE Rating",
        "ECBC Status",
        "Annual Carbon (t CO₂)",
        "CO₂ / Occupant (t/person)",
        "Avoidable Waste (MWh)",
        "Potential Savings (₹ Lakh)",
    ]

    for col in expected_columns:
        assert col in df.columns, f"Missing column: {col}"

    assert len(df) == 4
    assert set(df["Facility Type"]) == {"university", "hospital", "govt_office", "industrial"}

    for _, row in df.iterrows():
        assert row["Floor Area (m²)"] > 0
        assert row["Peak Occupancy"] > 0
        assert row["Annual Energy (MWh)"] > 0
        assert row["EPI (kWh/m²/yr)"] > 0
        assert row["ECBC Baseline"] > 0
        assert row["ECBC Status"] in ["Compliant", "Non-Compliant"]
        assert "⭐" in row["BEE Rating"]
        assert row["Annual Carbon (t CO₂)"] > 0
        assert row["CO₂ / Occupant (t/person)"] > 0


def test_calculate_state_impact_math():
    mock_df = pd.DataFrame({
        "Avoidable Waste (MWh)": [10.0, 30.0],          # mean = 20.0 MWh
        "Potential Savings (₹ Lakh)": [2.0, 6.0],        # mean = 4.0 Lakh
        "Avoidable Carbon (t CO₂)": [8.2, 24.6],         # mean = 16.4 t CO2
        "Annual Energy (MWh)": [500.0, 1000.0],
    })

    impact = calculate_state_impact(mock_df, num_facilities=100)

    # 20.0 MWh * 100 = 2000.0 MWh
    assert impact["total_annual_mwh_saved"] == 2000.0
    # 4.0 Lakh * 100 = 400.0 Lakh = 4.0 Crores
    assert impact["total_financial_savings_cr"] == 4.0
    # 16.4 tons * 100 = 1640.0 tons
    assert impact["total_co2_mitigated_tons"] == 1640.0
    # 1640.0 * 1000 / 21 = 78095
    assert impact["trees_planted_equivalent"] == int(1640.0 * 1000 / 21)
    # 2000.0 * 0.45 = 900.0 tons of coal
    assert impact["coal_unburned_tons"] == 900.0


def test_calculate_state_impact_empty():
    empty_df = pd.DataFrame()
    impact = calculate_state_impact(empty_df, num_facilities=100)
    assert impact["total_annual_mwh_saved"] == 0.0
    assert impact["total_financial_savings_cr"] == 0.0
    assert impact["total_co2_mitigated_tons"] == 0.0
    assert impact["trees_planted_equivalent"] == 0
    assert impact["coal_unburned_tons"] == 0.0
