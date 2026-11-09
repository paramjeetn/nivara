"""Unit tests for Service-Impact Guard (Step 7).

Validates institutional service protection constraints:
  - Hospital ICU & life-safety non-curtailability
  - Vaccine cold-chain refrigeration non-curtailability
  - ASHRAE 55 PMV thermal comfort boundary enforcement
  - Safe setpoint adjustment clearance
  - Off-hours lighting shutdown in unoccupied zones
  - In-hours lighting shutdown prevention in occupied zones
  - Batch partitioning of recommendations via apply_service_guard
"""
import pytest

from campus_energy.decision.guard import apply_service_guard, check_service_guard
from campus_energy.decision.recommendation import (
    GuardResult,
    Recommendation,
    SavingsBundle,
    SavingsRange,
)
from campus_energy.profiles import load_all_profiles, FacilityProfile


@pytest.fixture
def facility_profiles():
    """Load standard profiles from default profile directory."""
    return load_all_profiles("config/facility_profiles")


@pytest.fixture
def hospital_profile(facility_profiles) -> FacilityProfile:
    return facility_profiles["district_hospital"]


@pytest.fixture
def university_profile(facility_profiles) -> FacilityProfile:
    return facility_profiles["university"]


def test_hospital_icu_curtailment_blocked(hospital_profile):
    """Test 1: Hospital ICU HVAC or Life Support curtailment is blocked with safe alternative."""
    # Test 1a: ICU Life Support equipment
    rec_life_support = Recommendation(
        id="REC-ICU-EQ-001",
        facility_id="district_hospital",
        type="peak_shaving",
        target="EQ-IPD-ICU-EQ",
        action_text="Curtail ICU life support and monitoring equipment power by 35% during peak tariff hours.",
        trigger="peak_demand_spike",
        savings=SavingsBundle.from_expected(40.0),
        cost=0.0,
        owner_role="operations",
    )
    result_ls = check_service_guard(rec_life_support, hospital_profile)
    assert result_ls.status == "blocked"
    assert "protected_load_check" in result_ls.constraints_checked
    assert "protected" in result_ls.reason.lower() or "icu" in result_ls.reason.lower()
    assert result_ls.safe_alternative is not None
    assert len(result_ls.safe_alternative) > 15
    assert (
        "life-critical" in result_ls.safe_alternative.lower()
        or "uncurtailed" in result_ls.safe_alternative.lower()
        or "audit" in result_ls.safe_alternative.lower()
    )

    # Test 1b: ICU Precision HVAC unit
    rec_icu_hvac = Recommendation(
        id="REC-ICU-AC-001",
        facility_id="district_hospital",
        type="equipment_shutdown",
        target="EQ-IPD-ICU-AC",
        action_text="Shut down ICU precision HVAC unit for 45 minutes to shave facility peak demand.",
        trigger="peak_tariff_event",
        savings=SavingsBundle.from_expected(65.0),
        cost=0.0,
        owner_role="operations",
    )
    result_hvac = check_service_guard(rec_icu_hvac, hospital_profile)
    assert result_hvac.status == "blocked"
    assert "protected_load_check" in result_hvac.constraints_checked
    assert result_hvac.safe_alternative is not None


def test_vaccine_cold_chain_curtailment_blocked(hospital_profile):
    """Test 2: Vaccine cold-chain curtailment is blocked with actionable safe alternative."""
    rec_cold_chain = Recommendation(
        id="REC-CCR-001",
        facility_id="district_hospital",
        type="equipment_shutdown",
        target="EQ-IPD-CCR-AC",
        action_text="Turn off vaccine cold-chain refrigeration compressor for 2 hours during peak tariff window.",
        trigger="peak_shaving_window",
        savings=SavingsBundle.from_expected(28.0),
        cost=0.0,
        owner_role="operations",
    )
    result = check_service_guard(rec_cold_chain, hospital_profile)
    assert result.status == "blocked"
    assert "protected_load_check" in result.constraints_checked
    assert (
        "cold_chain" in result.reason.lower()
        or "vaccine" in result.reason.lower()
        or "refrigeration" in result.reason.lower()
    )
    assert result.safe_alternative is not None
    assert (
        "vaccine" in result.safe_alternative.lower()
        or "cold-chain" in result.safe_alternative.lower()
        or "uninterrupted" in result.safe_alternative.lower()
    )


def test_classroom_ac_setpoint_rise_pmv_blocked(university_profile):
    """Test 3: Classroom AC setpoint rise that causes PMV to exceed +0.5 is blocked with safe alternative."""
    # Proposing 28.0°C in classroom during lecture hours (optimal is 24°C, classroom max PMV is +0.5)
    rec_excess_temp = Recommendation(
        id="REC-SET-001",
        facility_id="university",
        type="setpoint_adjustment",
        target="BLD-A-CR101",
        action_text="Raise classroom AC setpoint to 28°C to reduce cooling chiller power.",
        trigger="high_ambient_temp_forecast",
        savings=SavingsBundle.from_expected(55.0),
        cost=0.0,
        owner_role="operations",
    )
    context = {
        "zone_id": "BLD-A-CR101",
        "proposed_temp_c": 28.0,
        "occupancy": 40,
        "scheduled_occupancy": 45,
        "humidity_pct": 50.0,
    }
    result = check_service_guard(rec_excess_temp, university_profile, context=context)
    assert result.status == "blocked"
    assert "pmv_comfort_check" in result.constraints_checked
    assert "pmv" in result.reason.lower() or "comfort" in result.reason.lower()
    assert result.safe_alternative is not None
    assert (
        "+1.0" in result.safe_alternative
        or "envelope" in result.safe_alternative
        or "pre-cool" in result.safe_alternative
    )


def test_safe_setpoint_rise_passes(university_profile):
    """Test 4: Safe setpoint rise (keeping PMV in bounds) passes all guard checks."""
    # Modest setpoint adjustment to 24.5°C in Classroom (within ±0.5 PMV bound)
    rec_safe_temp = Recommendation(
        id="REC-SET-002",
        facility_id="university",
        type="setpoint_adjustment",
        target="BLD-A-CR101",
        action_text="Adjust classroom AC setpoint to 24.5°C for calibrated energy optimization.",
        trigger="efficiency_baseline_opportunity",
        savings=SavingsBundle.from_expected(18.0),
        cost=0.0,
        owner_role="operations",
    )
    context = {
        "zone_id": "BLD-A-CR101",
        "proposed_temp_c": 24.5,
        "occupancy": 35,
        "scheduled_occupancy": 40,
        "humidity_pct": 50.0,
    }
    result = check_service_guard(rec_safe_temp, university_profile, context=context)
    assert result.status == "pass"
    assert "pmv_comfort_check" in result.constraints_checked
    assert "protected_load_check" in result.constraints_checked
    assert result.safe_alternative is None
    assert "satisfied" in result.reason.lower()


def test_off_hours_lighting_shutoff_unoccupied_passes(university_profile):
    """Test 5: Off-hours lighting shut-off in unoccupied zone passes."""
    rec_off_hours = Recommendation(
        id="REC-LT-OFF-001",
        facility_id="university",
        type="equipment_shutdown",
        target="BLD-A-CR101-LT",
        action_text="Shut off lighting in Classroom 101 during unoccupied off-hours after 22:00.",
        trigger="overnight_excess_anomaly",
        savings=SavingsBundle.from_expected(14.0),
        cost=0.0,
        owner_role="operations",
    )
    context = {
        "occupancy": 0,
        "scheduled_occupancy": 0,
        "is_occupied": False,
    }
    result = check_service_guard(rec_off_hours, university_profile, context=context)
    assert result.status == "pass"
    assert "occupancy_check" in result.constraints_checked
    assert result.safe_alternative is None


def test_in_hours_lighting_shutoff_occupied_blocked(university_profile):
    """Test 6: In-hours lighting shut-off in occupied zone is blocked with dimming alternative."""
    rec_in_hours = Recommendation(
        id="REC-LT-IN-001",
        facility_id="university",
        type="lighting_curtailment",
        target="BLD-A-CR101-LT",
        action_text="Turn off primary lighting in Classroom 101 during active class hours to shave demand.",
        trigger="demand_limit_exceeded",
        savings=SavingsBundle.from_expected(22.0),
        cost=0.0,
        owner_role="operations",
    )
    context = {
        "occupancy": 45,
        "scheduled_occupancy": 50,
        "is_occupied": True,
    }
    result = check_service_guard(rec_in_hours, university_profile, context=context)
    assert result.status == "blocked"
    assert "occupancy_check" in result.constraints_checked
    assert "occupancy" in result.reason.lower() or "service level" in result.reason.lower()
    assert result.safe_alternative is not None
    assert (
        "dim" in result.safe_alternative.lower()
        or "70%" in result.safe_alternative
        or "night setback" in result.safe_alternative.lower()
    )


def test_apply_service_guard_partitions_mixed_batch(hospital_profile, university_profile):
    """Test 7: apply_service_guard correctly partitions a mixed list of recommendations into approved and blocked."""
    # 3 Blocked Recommendations
    rec_icu = Recommendation(
        id="REC-BLK-001",
        facility_id="district_hospital",
        type="peak_shaving",
        target="EQ-IPD-ICU-EQ",
        action_text="Curtail ICU ventilator power during peak grid tariff event.",
        trigger="peak_event",
        savings=SavingsBundle.from_expected(50.0),
        owner_role="operations",
    )
    rec_vaccine = Recommendation(
        id="REC-BLK-002",
        facility_id="district_hospital",
        type="equipment_shutdown",
        target="EQ-IPD-CCR-AC",
        action_text="Shut down cold chain vaccine refrigeration unit to reduce demand charge.",
        trigger="demand_charge_warning",
        savings=SavingsBundle.from_expected(35.0),
        owner_role="operations",
    )
    rec_hot_ac = Recommendation(
        id="REC-BLK-003",
        facility_id="district_hospital",
        type="setpoint_adjustment",
        target="EQ-IPD-GW-AC",
        action_text="Raise General Ward AC setpoint to 29°C during occupied visiting hours.",
        trigger="high_load",
        savings=SavingsBundle.from_expected(45.0),
        owner_role="operations",
    )

    # 2 Approved Recommendations
    rec_safe_ac = Recommendation(
        id="REC-APP-001",
        facility_id="district_hospital",
        type="setpoint_adjustment",
        target="EQ-IPD-GW-AC",
        action_text="Adjust General Ward AC setpoint to 24.5°C to optimize chiller efficiency.",
        trigger="scheduled_optimization",
        savings=SavingsBundle.from_expected(16.0),
        owner_role="operations",
    )
    rec_off_hours_lt = Recommendation(
        id="REC-APP-002",
        facility_id="district_hospital",
        type="equipment_shutdown",
        target="EQ-ADM-OF-LT",
        action_text="Shut off lighting in administrative offices during unoccupied off-hours after 22:00.",
        trigger="unoccupied_schedule",
        savings=SavingsBundle.from_expected(12.0),
        owner_role="operations",
    )

    batch = [rec_icu, rec_safe_ac, rec_vaccine, rec_off_hours_lt, rec_hot_ac]
    context = {
        "REC-BLK-003": {"proposed_temp_c": 29.0, "occupancy": 30},
        "REC-APP-001": {"proposed_temp_c": 24.5, "occupancy": 30},
        "REC-APP-002": {"occupancy": 0, "scheduled_occupancy": 0},
    }

    approved, blocked = apply_service_guard(batch, hospital_profile, context=context)

    assert len(approved) == 2
    assert len(blocked) == 3

    assert rec_safe_ac in approved
    assert rec_off_hours_lt in approved
    assert rec_icu in blocked
    assert rec_vaccine in blocked
    assert rec_hot_ac in blocked

    for app in approved:
        assert app.is_passed is True
        assert app.is_blocked is False
        assert app.service_guard.status == "pass"
        assert app.service_guard.safe_alternative is None

    for blk in blocked:
        assert blk.is_blocked is True
        assert blk.is_passed is False
        assert blk.service_guard.status == "blocked"
        assert blk.service_guard.safe_alternative is not None
        assert len(blk.service_guard.constraints_checked) > 0
