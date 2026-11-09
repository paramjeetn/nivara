"""Unit tests for Facility Profiles and Protected Loads (Step 6)."""
from pathlib import Path
import pytest
import yaml

from campus_energy.profiles import (
    ComfortBound,
    ProtectedLoad,
    FacilityProfile,
    load_facility_profile,
    load_all_profiles,
)


def test_load_all_standard_profiles():
    """Verify loading all standard facility profiles from default directory."""
    profiles = load_all_profiles("config/facility_profiles")
    assert len(profiles) >= 4

    expected_ids = ["university", "district_hospital", "government_office", "industrial_campus"]
    for fid in expected_ids:
        assert fid in profiles
        prof = profiles[fid]
        assert isinstance(prof, FacilityProfile)
        assert prof.facility_id == fid
        assert prof.floor_area_m2 > 0
        assert prof.peak_occupancy > 0
        assert len(prof.protected_loads) > 0
        assert len(prof.comfort_bounds) > 0
        assert len(prof.compliance_frameworks) > 0
        assert len(prof.default_roles) > 0


def test_load_individual_profiles():
    """Verify loading each YAML directly via load_facility_profile."""
    for slug in ["university", "district_hospital", "government_office", "industrial_campus"]:
        path = f"config/facility_profiles/{slug}.yaml"
        profile = load_facility_profile(path)
        assert profile.facility_id == slug
        assert isinstance(profile.floor_area_m2, float)
        assert isinstance(profile.peak_occupancy, int)
        assert profile.tariff_profile != ""


def test_hospital_critical_protected_loads():
    """Verify hospital profile contains non-curtailable ICU, OT, and cold-chain loads."""
    profiles = load_all_profiles("config/facility_profiles")
    hospital = profiles["district_hospital"]

    # Verify hospital profile identification
    assert hospital.facility_type == "hospital"
    assert hospital.facility_name == "Rajiv Gandhi District Hospital"

    # Check protected loads
    protected_asset_ids = {load.asset_id: load for load in hospital.protected_loads}

    # ICU life support and precision HVAC
    assert "EQ-IPD-ICU-EQ" in protected_asset_ids
    icu_eq = protected_asset_ids["EQ-IPD-ICU-EQ"]
    assert icu_eq.curtailable is False
    assert icu_eq.category == "life_safety"
    assert icu_eq.min_power_kw is not None and icu_eq.min_power_kw > 0
    assert "ventilator" in icu_eq.reason.lower() or "monitoring" in icu_eq.reason.lower()

    assert "EQ-IPD-ICU-AC" in protected_asset_ids
    icu_ac = protected_asset_ids["EQ-IPD-ICU-AC"]
    assert icu_ac.curtailable is False

    # OT surgical equipment, lights, HVAC
    assert "EQ-OT-OT1-EQ" in protected_asset_ids
    ot_eq = protected_asset_ids["EQ-OT-OT1-EQ"]
    assert ot_eq.curtailable is False
    assert ot_eq.category == "life_safety"

    assert "EQ-OT-OT1-LT" in protected_asset_ids
    ot_lt = protected_asset_ids["EQ-OT-OT1-LT"]
    assert ot_lt.curtailable is False

    assert "EQ-OT-OT1-AC" in protected_asset_ids
    ot_ac = protected_asset_ids["EQ-OT-OT1-AC"]
    assert ot_ac.curtailable is False

    # Cold chain refrigeration (vaccines / blood bank)
    assert "EQ-IPD-CCR-AC" in protected_asset_ids
    cold_chain = protected_asset_ids["EQ-IPD-CCR-AC"]
    assert cold_chain.curtailable is False
    assert cold_chain.category == "cold_chain"
    assert "vaccine" in cold_chain.reason.lower() or "blood" in cold_chain.reason.lower()

    # All protected loads in hospital must be strictly non-curtailable
    for load in hospital.protected_loads:
        assert load.curtailable is False, f"Load {load.asset_id} unexpectedly curtailable"


def test_is_asset_protected_semantics():
    """Verify is_asset_protected and get_protected_load methods."""
    profiles = load_all_profiles("config/facility_profiles")
    hospital = profiles["district_hospital"]
    university = profiles["university"]

    # Protected asset checks
    assert hospital.is_asset_protected("EQ-IPD-ICU-EQ") is True
    assert hospital.is_asset_protected("EQ-IPD-CCR-AC") is True
    assert hospital.is_asset_protected("EQ-OT-OT1-AC") is True

    # Non-protected asset checks (e.g. general waiting room lights, corridors)
    assert hospital.is_asset_protected("EQ-OPD-WR1-LT") is False
    assert hospital.is_asset_protected("NON_EXISTENT_ASSET_123") is False

    # Retrieve ProtectedLoad record
    rec = hospital.get_protected_load("EQ-IPD-ICU-EQ")
    assert rec is not None
    assert rec.asset_name == "ICU Life Support & Monitors"
    assert rec.curtailable is False

    # University checks
    assert university.is_asset_protected("EQ-A-CL01-PC") is True
    assert university.is_asset_protected("EQ-B-LAB1-EQ") is True
    assert university.is_asset_protected("EQ-A-CR101-LT") is False  # General classroom lights


def test_comfort_bounds_and_fallbacks():
    """Verify comfort bound retrieval and zone matching."""
    profiles = load_all_profiles("config/facility_profiles")
    hospital = profiles["district_hospital"]
    university = profiles["university"]

    # Hospital ICU has strict ±0.2 PMV bounds
    icu_bounds = hospital.get_comfort_bound_for_zone("icu")
    assert icu_bounds.min_pmv == -0.2
    assert icu_bounds.max_pmv == 0.2

    # Hospital wards have ±0.3 PMV bounds
    ward_bounds = hospital.get_comfort_bound_for_zone("wards")
    assert ward_bounds.min_pmv == -0.3
    assert ward_bounds.max_pmv == 0.3

    # Hospital OPD has standard ±0.5 PMV bounds
    opd_bounds = hospital.get_comfort_bound_for_zone("opd")
    assert opd_bounds.min_pmv == -0.5
    assert opd_bounds.max_pmv == 0.5

    # University classrooms
    cr_bounds = university.get_comfort_bound_for_zone("classrooms")
    assert cr_bounds.min_pmv == -0.5
    assert cr_bounds.max_pmv == 0.5

    # Smart alias / substring matching
    cr_alias_bounds = university.get_comfort_bound_for_zone("BLD-A-CR101")
    assert cr_alias_bounds.min_pmv == -0.5
    assert cr_alias_bounds.max_pmv == 0.5

    # Unknown zone fallback
    fallback_bounds = hospital.get_comfort_bound_for_zone("unknown_unmapped_zone")
    assert isinstance(fallback_bounds, ComfortBound)
    assert fallback_bounds.min_pmv <= fallback_bounds.max_pmv


def test_synthetic_fifth_facility_profile(tmp_path: Path):
    """Verify an entirely new facility type loads without any code modifications."""
    dc_yaml_content = {
        "facility_id": "tier3_data_center",
        "facility_name": "Odisha Cloud Data Centre",
        "facility_type": "data_center",
        "location": "Infovalley, Bhubaneswar, Odisha",
        "floor_area_m2": 4200.0,
        "peak_occupancy": 65,
        "data_dir": "data/data_center",
        "icon": "🖥️",
        "tariff_profile": "odisha_cesu_ht_continuous",
        "compliance_frameworks": [
            "TIA-942 Tier III",
            "ASHRAE TC 9.9 Thermal Guidelines",
            "ISO 27001",
            "BEE Star Rating",
        ],
        "default_roles": ["data_center_manager", "facilities_engineer", "security_officer"],
        "comfort_bounds": {
            "server_hall": {
                "min_pmv": -0.2,
                "max_pmv": 0.2,
                "temp_min_c": 18.0,
                "temp_max_c": 22.0,
            },
            "noc": {
                "min_pmv": -0.4,
                "max_pmv": 0.4,
                "temp_min_c": 21.0,
                "temp_max_c": 24.0,
            },
            "general": {
                "min_pmv": -0.5,
                "max_pmv": 0.5,
                "temp_min_c": 22.0,
                "temp_max_c": 26.0,
            },
        },
        "protected_loads": [
            {
                "asset_id": "EQ-DC-UPS-01",
                "asset_name": "Modular Dual-Bus Online UPS 500kVA",
                "zone_id": "BLD-DC-UPS",
                "category": "critical_it",
                "curtailable": False,
                "min_power_kw": 80.0,
                "reason": "Uninterruptible power delivery to Tier III server racks; zero transfer time required.",
            },
            {
                "asset_id": "EQ-DC-CRAC-01",
                "asset_name": "Precision Computer Room Air Conditioning (CRAC) Unit 1",
                "zone_id": "BLD-DC-HALL1",
                "category": "life_safety",
                "curtailable": False,
                "min_power_kw": 25.0,
                "reason": "Thermal envelope protection against server CPU thermal throttling or thermal trip.",
            },
        ],
    }

    yaml_file = tmp_path / "data_center.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(dc_yaml_content, f)

    # 1. Load single profile directly from arbitrary temp path
    dc_profile = load_facility_profile(yaml_file)
    assert dc_profile.facility_id == "tier3_data_center"
    assert dc_profile.facility_name == "Odisha Cloud Data Centre"
    assert dc_profile.facility_type == "data_center"
    assert dc_profile.floor_area_m2 == 4200.0
    assert dc_profile.is_asset_protected("EQ-DC-UPS-01") is True
    assert dc_profile.is_asset_protected("EQ-DC-CRAC-01") is True
    assert dc_profile.is_asset_protected("EQ-LIGHTS-OFFICE") is False

    # 2. Discover via load_all_profiles from the temp directory
    discovered = load_all_profiles(tmp_path)
    assert len(discovered) == 1
    assert "tier3_data_center" in discovered
    assert discovered["tier3_data_center"].facility_type == "data_center"


def test_facility_profile_dict_backwards_compatibility():
    """Verify FacilityProfile supports subscripting and get() for backwards compatibility."""
    profiles = load_all_profiles("config/facility_profiles")
    profile = profiles["university"]

    assert profile["key"] == "university"
    assert profile["facility_type"] == "university"
    assert profile["floor_area_m2"] == 12000.0
    assert profile["peak_occupancy"] == 2500
    assert isinstance(profile["data_dir"], Path)
    assert profile.get("location") == "Bhubaneswar, Odisha"
    assert profile.get("non_existent_key", "default_val") == "default_val"
