"""Unit and Property Tests for India-Specific Levers and Tariffs.

Tests:
  1. Indian tariff configuration loading, fallback, and ToD band evaluation.
  2. Maximum Demand (kVA) peak shaving calculations and recommendation generation.
  3. Power Factor (PF) penalty surcharge and APFC retrofit sizing / economics.
  4. Time-of-Day (ToD) deferrable pump load shifting and rate differential savings.
  5. Service-Impact Guard enforcement preventing curtailment of mission-critical / hospital ICU loads.
  6. Diesel Generator (DG) operational compliance audit and roadmap flagging.
  7. Integration with Unified Recommendation Engine (`build_ranked_recommendations`).
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from campus_energy.decision.guard import apply_service_guard, check_service_guard
from campus_energy.decision.india_levers import (
    audit_diesel_generators,
    calculate_apfc_requirements,
    calculate_telemetry_kva,
    generate_dg_audit_recommendation,
    generate_india_tariff_recommendations,
    generate_peak_demand_recommendation,
    generate_power_factor_recommendation,
    generate_tod_load_shifting_recommendations,
)
from campus_energy.decision.recommendation import Recommendation, SavingsBundle
from campus_energy.decision.engine import RecommendationEngine, build_ranked_recommendations
from campus_energy.profiles import FacilityProfile, ProtectedLoad, load_facility_profile
from campus_energy.tariffs import DEFAULT_TARIFF_ID, IndianTariff, ToDBand, load_tariff


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def odisha_tariff() -> IndianTariff:
    """Load canonical TPCODL / OERC HT Commercial tariff."""
    return load_tariff("odisha_ht_commercial")


@pytest.fixture
def sample_telemetry() -> pd.DataFrame:
    """Generate 48 hours of realistic 15-minute interval campus telemetry."""
    # 48 hours at 15-min intervals = 192 intervals
    timestamps = pd.date_range("2026-09-01 00:00:00", periods=192, freq="15min", tz="Asia/Kolkata")
    records = []

    for ts in timestamps:
        hour = ts.hour
        # Base active power (kW) higher during day and evening peak (18:00 - 22:00)
        if 18 <= hour < 22:
            p_kw = 420.0 + (ts.minute % 15) * 5.0  # Peak evening demand
        elif 9 <= hour < 18:
            p_kw = 350.0
        else:
            p_kw = 120.0  # Night baseline

        # Main meter record with uncorrected low PF (0.84)
        records.append({
            "reading_id": f"R-{len(records)}",
            "timestamp": ts.isoformat(),
            "asset_id": "MTR-BLD-MAIN",
            "reading_type": "meter",
            "power_kw": p_kw,
            "interval_energy_kwh": p_kw * 0.25,
            "power_factor": 0.84,  # Triggers low PF penalty (< 0.90)
        })

        # Water transfer pump operating during peak hours (18:00 - 21:00)
        pump_power = 11.0 if 18 <= hour < 21 else 0.0
        records.append({
            "reading_id": f"R-{len(records)}",
            "timestamp": ts.isoformat(),
            "asset_id": "EQ-UTL-P1-PM",
            "reading_type": "equipment",
            "power_kw": pump_power,
            "interval_energy_kwh": pump_power * 0.25,
            "power_factor": 0.85,
        })

    return pd.DataFrame(records)


@pytest.fixture
def hospital_profile() -> FacilityProfile:
    """Load or synthesize a District Hospital profile with non-curtailable ICU loads."""
    try:
        return load_facility_profile("district_hospital")
    except Exception:
        return FacilityProfile(
            facility_id="district_hospital",
            facility_name="Rajiv Gandhi District Hospital",
            facility_type="hospital",
            location="Cuttack, Odisha",
            floor_area_m2=8500.0,
            peak_occupancy=420,
            data_dir="data/hospital",
            tariff_profile="odisha_ht_commercial",
            protected_loads=[
                ProtectedLoad(
                    asset_id="EQ-IPD-ICU-EQ",
                    asset_name="ICU Life Support & Monitors",
                    zone_id="BLD-IPD-ICU",
                    category="life_safety",
                    curtailable=False,
                    min_power_kw=6.0,
                    reason="Continuous patient hemodynamics, mechanical ventilators, and life-critical cardiac monitoring.",
                )
            ],
        )


@pytest.fixture
def university_profile() -> FacilityProfile:
    """Load or synthesize University profile."""
    try:
        return load_facility_profile("university")
    except Exception:
        return FacilityProfile(
            facility_id="university",
            facility_name="Vidya Institute of Technology",
            facility_type="university",
            location="Bhubaneswar, Odisha",
            floor_area_m2=12000.0,
            peak_occupancy=2500,
            data_dir="data/generated",
            tariff_profile="odisha_ht_commercial",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tariff Loading & Calculation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_tariff_loading_and_fallback():
    """Verify tariff loader parses YAML accurately and handles nonexistent files gracefully."""
    tariff = load_tariff("odisha_ht_commercial")
    assert tariff.tariff_id == "odisha_ht_commercial"
    assert tariff.energy_rate_inr_per_kwh == 8.50
    assert tariff.demand_charge_inr_per_kva_month == 250.0
    assert tariff.contract_demand_kva == 450.0
    assert tariff.billing_demand_percent == 80.0
    assert tariff.pf_penalty_threshold == 0.90
    assert tariff.pf_incentive_threshold == 0.95
    assert "Illustrative tariff" in tariff.disclaimer

    # Nonexistent tariff ID should fallback safely without raising exceptions
    fallback = load_tariff("nonexistent_tariff_xyz")
    assert isinstance(fallback, IndianTariff)
    assert fallback.tariff_id == "nonexistent_tariff_xyz"
    assert fallback.energy_rate_inr_per_kwh == 8.50


def test_tariff_tod_bands(odisha_tariff: IndianTariff):
    """Verify Time-of-Day pricing bands and multipliers."""
    # Peak: 18:00 - 22:00 -> ₹10.20/kWh (1.20x)
    for h in [18, 19, 20, 21]:
        band_name, band = odisha_tariff.get_tod_band_for_hour(h)
        assert band_name == "peak"
        assert band.rate_inr == 10.20
        assert odisha_tariff.get_rate_for_hour(h) == 10.20

    # Off-peak: 00:00 - 06:00 -> ₹7.225/kWh (0.85x)
    for h in [0, 1, 2, 3, 4, 5]:
        band_name, band = odisha_tariff.get_tod_band_for_hour(h)
        assert band_name == "off_peak"
        assert band.rate_inr == 7.225
        assert odisha_tariff.get_rate_for_hour(h) == 7.225

    # Normal: 06:00 - 18:00, 22:00 - 24:00 -> ₹8.50/kWh (1.0x)
    for h in [6, 10, 14, 17, 22, 23]:
        band_name, band = odisha_tariff.get_tod_band_for_hour(h)
        assert band_name == "normal"
        assert band.rate_inr == 8.50


def test_billing_demand_calculation(odisha_tariff: IndianTariff):
    """Verify billing demand = max(recorded_kva, 80% contract_demand)."""
    cd = 450.0
    min_billing_demand = 450.0 * 0.80  # 360.0 kVA

    # Recorded demand below 80% threshold -> billed on 360 kVA
    assert odisha_tariff.calculate_billing_demand(300.0, contract_demand_override=cd) == 360.0
    assert odisha_tariff.calculate_demand_charge(300.0, contract_demand_override=cd) == 360.0 * 250.0

    # Recorded demand above threshold -> billed on actual recorded kVA
    assert odisha_tariff.calculate_billing_demand(400.0, contract_demand_override=cd) == 400.0
    assert odisha_tariff.calculate_demand_charge(400.0, contract_demand_override=cd) == 400.0 * 250.0

    # Recorded demand exceeding contract demand -> billed on actual recorded kVA
    assert odisha_tariff.calculate_billing_demand(480.0, contract_demand_override=cd) == 480.0
    assert odisha_tariff.calculate_demand_charge(480.0, contract_demand_override=cd) == 480.0 * 250.0


def test_power_factor_penalties_and_incentives(odisha_tariff: IndianTariff):
    """Verify OERC PF penalty (1% per 0.01 drop < 0.90) and incentive (0.5% per 0.01 rise > 0.95)."""
    energy_charges = 100_000.0  # ₹1,00,000

    # 1. PF = 0.85 -> 0.05 drop below 0.90 -> 5% penalty = ₹5,000
    adj, status, pct = odisha_tariff.calculate_pf_adjustment(0.85, energy_charges)
    assert status == "penalty"
    assert pct == 5.0
    assert adj == 5_000.0

    # 2. PF = 0.80 -> 0.10 drop below 0.90 -> 10% penalty = ₹10,000
    adj, status, pct = odisha_tariff.calculate_pf_adjustment(0.80, energy_charges)
    assert status == "penalty"
    assert pct == 10.0
    assert adj == 10_000.0

    # 3. PF = 0.92 -> compliant normal band (0.90 <= PF <= 0.95) -> 0 adjustment
    adj, status, pct = odisha_tariff.calculate_pf_adjustment(0.92, energy_charges)
    assert status == "normal"
    assert pct == 0.0
    assert adj == 0.0

    # 4. PF = 0.98 -> 0.03 rise above 0.95 -> 3 * 0.5% = 1.5% incentive credit = -₹1,500
    adj, status, pct = odisha_tariff.calculate_pf_adjustment(0.98, energy_charges)
    assert status == "incentive"
    assert pct == 1.5
    assert adj == -1_500.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Maximum Demand (kVA) Peak Shaving Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_telemetry_kva_calculation():
    """Verify kVA = kW / PF accurately computed and clamped against zero division."""
    df = pd.DataFrame({
        "power_kw": [100.0, 200.0, 0.0],
        "power_factor": [0.80, 0.85, 0.0],
    })
    res = calculate_telemetry_kva(df)
    assert "power_kva" in res.columns
    assert res.iloc[0]["power_kva"] == pytest.approx(125.0, 0.1)  # 100 / 0.80
    assert res.iloc[1]["power_kva"] == pytest.approx(235.29, 0.1)  # 200 / 0.85
    assert res.iloc[2]["power_kva"] == 0.0  # 0 kW handled safely


def test_peak_demand_shaving_recommendation(sample_telemetry: pd.DataFrame, university_profile: FacilityProfile):
    """Verify peak kVA detection and BESS peak demand shaving recommendation generation."""
    tariff = load_tariff("odisha_ht_commercial")
    rec = generate_peak_demand_recommendation(
        telemetry_df=sample_telemetry,
        profile=university_profile,
        tariff=tariff,
        contract_demand_kva=450.0,
    )

    assert rec is not None
    assert rec.type == "peak_demand_shaving"
    assert "maximum demand shaving" in rec.action_text.lower()
    assert rec.savings.inr.expected > 0.0
    assert rec.cost > 0.0  # BESS capex modeled
    assert rec.payback_months is not None and rec.payback_months > 0.0
    assert "tariff.demand_charge_inr_per_kva_month" in rec.assumption_refs


# ─────────────────────────────────────────────────────────────────────────────
# 3. Power Factor (PF) & APFC Retrofit Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_apfc_kvar_calculation():
    """Verify reactive power formula: kVAR = P * (tan(phi_1) - tan(phi_2))."""
    p_kw = 100.0
    pf_current = 0.80
    pf_target = 0.98

    # phi_1 = arccos(0.80) -> tan(phi_1) = 0.75
    # phi_2 = arccos(0.98) -> tan(phi_2) = sqrt(1 - 0.98^2)/0.98 = 0.20306
    # Delta kVAR = 100 * (0.75 - 0.20306) = 54.69 kVAR -> rounded to ~55 kVAR
    kvar = calculate_apfc_requirements(p_kw, pf_current, pf_target)
    assert kvar >= 50.0
    assert kvar <= 60.0

    # If already at or above target PF, required kVAR is 0
    assert calculate_apfc_requirements(100.0, 0.98, 0.98) == 0.0
    assert calculate_apfc_requirements(100.0, 0.99, 0.98) == 0.0


def test_power_factor_recommendation_generation(sample_telemetry: pd.DataFrame, university_profile: FacilityProfile):
    """Verify APFC recommendation generated when telemetry PF < 0.90."""
    tariff = load_tariff("odisha_ht_commercial")
    rec = generate_power_factor_recommendation(
        telemetry_df=sample_telemetry,
        profile=university_profile,
        tariff=tariff,
        target_pf=0.98,
    )

    assert rec is not None
    assert rec.type == "power_factor_correction"
    assert "APFC" in rec.action_text
    assert "low-PF penalty surcharge" in rec.action_text
    assert rec.savings.inr.expected > 0.0
    assert rec.cost > 0.0
    assert rec.payback_months is not None and rec.payback_months > 0.0
    assert "tariff.pf_penalty_threshold" in rec.assumption_refs
    assert "capital_cost.apfc_per_kvar" in rec.assumption_refs


# ─────────────────────────────────────────────────────────────────────────────
# 4. Time-of-Day (ToD) Load Shifting Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_tod_load_shifting_recommendation(sample_telemetry: pd.DataFrame, university_profile: FacilityProfile):
    """Verify deferrable pump operating during peak window (18:00–22:00) gets shifted to off-peak night."""
    tariff = load_tariff("odisha_ht_commercial")
    recs = generate_tod_load_shifting_recommendations(
        telemetry_df=sample_telemetry,
        profile=university_profile,
        tariff=tariff,
    )

    assert len(recs) >= 1
    pump_rec = next((r for r in recs if r.target == "EQ-UTL-P1-PM"), None)
    assert pump_rec is not None
    assert pump_rec.type == "tod_load_shifting"
    assert "18:00–22:00" in pump_rec.action_text or "peak" in pump_rec.action_text.lower()
    assert "01:00–05:00" in pump_rec.action_text or "off-peak" in pump_rec.action_text.lower()
    assert pump_rec.savings.inr.expected > 0.0
    assert pump_rec.cost == 0.0  # Operational reschedule
    assert pump_rec.payback_months == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Service-Impact Guard Critical Load Protection
# ─────────────────────────────────────────────────────────────────────────────

def test_service_guard_blocks_hospital_icu_curtailment(hospital_profile: FacilityProfile):
    """Verify Service Guard strictly BLOCKS peak shaving or load shifting on hospital ICU life support."""
    bundle = SavingsBundle.from_expected(100.0)

    # Malicious or mistaken peak shaving action targeting ICU
    bad_rec = Recommendation(
        id="REC-TEST-ICU",
        facility_id=hospital_profile.facility_id,
        type="peak_demand_shaving",
        target="EQ-IPD-ICU-EQ",  # Protected ICU load
        action_text="Curtail ICU ventilator power during peak hours to avoid contract demand penalty.",
        trigger="peak_demand_exceeded",
        savings=bundle,
        cost=0.0,
    )

    guard_res = check_service_guard(bad_rec, hospital_profile)
    assert guard_res.status == "blocked"
    assert "protected mission-critical asset" in guard_res.reason.lower() or "life_safety" in guard_res.reason.lower()

    # Apply batch guard
    approved, blocked = apply_service_guard([bad_rec], hospital_profile)
    assert len(approved) == 0
    assert len(blocked) == 1
    assert blocked[0].id == "REC-TEST-ICU"
    assert blocked[0].is_blocked is True


def test_service_guard_permits_non_critical_pump_shifting(university_profile: FacilityProfile):
    """Verify Service Guard permits ToD shifting on non-protected utility pump."""
    bundle = SavingsBundle.from_expected(50.0)

    pump_rec = Recommendation(
        id="REC-TEST-PUMP",
        facility_id=university_profile.facility_id,
        type="tod_load_shifting",
        target="EQ-UTL-P1-PM",  # Water pump in university
        action_text="Shift water transfer pump to 01:00-05:00 off-peak night window.",
        trigger="tod_peak_savings",
        savings=bundle,
        cost=0.0,
    )

    guard_res = check_service_guard(pump_rec, university_profile)
    assert guard_res.status == "pass"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Diesel Generator (DG) Audit Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_diesel_generator_audit(hospital_profile: FacilityProfile):
    """Verify DG audit reports fuel price (₹95/L), emissions (2.68 kg/L), generation cost, and roadmap."""
    audit = audit_diesel_generators(profile=hospital_profile)
    assert audit["diesel_fuel_price_inr_per_l"] == 95.0
    assert audit["direct_emissions_kg_per_l"] == 2.68
    assert audit["specific_fuel_consumption_l_per_kwh"] == 0.30
    assert audit["dg_generation_cost_inr_per_kwh"] == pytest.approx(28.50, 0.1)
    assert audit["cost_multiplier_vs_grid"] > 2.5
    assert "roadmap" in audit["roadmap_feature"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Unified Recommendation Engine Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_unified_recommendation_engine_india_levers(
    sample_telemetry: pd.DataFrame, university_profile: FacilityProfile
):
    """Verify build_ranked_recommendations synthesizes, guards, and ranks India tariff recommendations."""
    approved, blocked = build_ranked_recommendations(
        facility_profile=university_profile,
        telemetry_df=sample_telemetry,
    )

    # Approved list should contain India recommendations sorted by expected INR savings descending
    assert len(approved) > 0
    rec_types = {r.type for r in approved}
    assert ("peak_demand_shaving" in rec_types) or ("power_factor_correction" in rec_types) or ("tod_load_shifting" in rec_types)

    # Check ranking monotonicity
    for i in range(len(approved) - 1):
        assert approved[i].savings.inr.expected >= approved[i + 1].savings.inr.expected

    # All approved items must have is_estimate=True and assumption_refs
    for r in approved:
        assert r.is_estimate is True
        assert len(r.assumption_refs) > 0
