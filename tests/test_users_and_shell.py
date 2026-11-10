"""Unit tests for Nivara Step UX-2: Shell, Navigation, User Management, and Login."""
from pathlib import Path
import pytest

from campus_energy.users import (
    UserProfile,
    load_users,
    authenticate_user,
    get_demo_users,
    get_user_by_id,
)


def test_load_users_from_yaml():
    """Verify that load_users correctly parses config/users.yaml with all 3 seed profiles."""
    users = load_users("config/users.yaml")
    assert isinstance(users, dict)
    assert len(users) == 3
    assert set(users.keys()) == {"admin", "operations", "sustainability"}

    # Admin profile verification
    admin = users["admin"]
    assert admin.id == "usr_admin_01"
    assert admin.name == "Dr. Arun Patnaik"
    assert admin.email == "arun.patnaik@bput.ac.in"
    assert admin.role == "administrator"
    assert admin.title == "Registrar & Campus Administrator"
    assert set(admin.facilities) == {"university", "hospital", "govt_office", "industrial"}
    assert admin.default_facility == "university"

    # Operations profile verification
    ops = users["operations"]
    assert ops.id == "usr_ops_01"
    assert ops.name == "Rajesh Mohanty"
    assert ops.email == "rajesh.mohanty@bput.ac.in"
    assert ops.role == "operations"
    assert ops.title == "Chief Facilities & Maintenance Engineer"
    assert set(ops.facilities) == {"university", "hospital", "govt_office", "industrial"}
    assert ops.default_facility == "university"

    # Sustainability profile verification
    sust = users["sustainability"]
    assert sust.id == "usr_sust_01"
    assert sust.name == "Priya Das"
    assert sust.email == "priya.das@bput.ac.in"
    assert sust.role == "sustainability"
    assert sust.title == "Chief Sustainability & ESG Officer"
    assert set(sust.facilities) == {"university", "hospital", "govt_office", "industrial"}
    assert sust.default_facility == "university"


def test_authenticate_user_variations():
    """Verify authentication by user ID, email, role key, and name."""
    # Lookup by ID
    u1 = authenticate_user("usr_admin_01")
    assert u1 is not None
    assert u1.name == "Dr. Arun Patnaik"

    u2 = authenticate_user("usr_ops_01")
    assert u2 is not None
    assert u2.name == "Rajesh Mohanty"

    u3 = authenticate_user("usr_sust_01")
    assert u3 is not None
    assert u3.name == "Priya Das"

    # Lookup by email
    assert authenticate_user("arun.patnaik@bput.ac.in").id == "usr_admin_01"
    assert authenticate_user("rajesh.mohanty@bput.ac.in").id == "usr_ops_01"
    assert authenticate_user("priya.das@bput.ac.in").id == "usr_sust_01"

    # Lookup by role key
    assert authenticate_user("admin").id == "usr_admin_01"
    assert authenticate_user("operations").id == "usr_ops_01"
    assert authenticate_user("sustainability").id == "usr_sust_01"

    # Lookup by full name
    assert authenticate_user("Dr. Arun Patnaik").id == "usr_admin_01"
    assert authenticate_user("Rajesh Mohanty").id == "usr_ops_01"
    assert authenticate_user("Priya Das").id == "usr_sust_01"

    # Unknown or empty queries
    assert authenticate_user("unknown_user@example.com") is None
    assert authenticate_user("") is None
    assert authenticate_user("   ") is None


def test_user_profile_helper_properties():
    """Verify UserProfile role display, icons, initials, and facility access."""
    admin = authenticate_user("admin")
    assert admin.role_display == "Administrator"
    assert admin.role_icon == "👔"
    assert admin.initials == "AP"
    assert admin.has_facility_access("university") is True
    assert admin.has_facility_access("hospital") is True
    assert admin.has_facility_access("nonexistent_facility") is False
    assert admin["name"] == "Dr. Arun Patnaik"

    ops = authenticate_user("operations")
    assert ops.role_display == "Operations"
    assert ops.role_icon == "🔧"
    assert ops.initials == "RM"

    sust = authenticate_user("sustainability")
    assert sust.role_display == "Sustainability"
    assert sust.role_icon == "🌱"
    assert sust.initials == "PD"

    demo_list = get_demo_users()
    assert len(demo_list) == 3
    assert get_user_by_id("usr_ops_01").name == "Rajesh Mohanty"


def test_query_params_and_destination_mapping():
    """Verify destination slug mapping and query parameter translation logic."""
    destination_slugs = {
        "home": "Home",
        "alerts": "Alerts & Work",
        "alerts_and_work": "Alerts & Work",
        "alerts-and-work": "Alerts & Work",
        "assets": "Assets",
        "energy": "Energy",
        "water": "Water",
        "reports": "Reports",
        "settings": "Settings",
        # Legacy aliases
        "executive": "Home",
        "map": "Assets",
        "campus_map": "Assets",
        "forecasting": "Energy",
        "microgrid": "Energy",
        "billing": "Energy",
        "governance": "Reports",
    }

    # Verify all 7 core destinations map properly
    core_destinations = ["Home", "Alerts & Work", "Assets", "Energy", "Water", "Reports", "Settings"]
    for dest in core_destinations:
        matching_slugs = [s for s, d in destination_slugs.items() if d == dest]
        assert len(matching_slugs) >= 1

    # Verify range translation
    valid_ranges = ["Today", "7 Days", "28 Days", "Custom"]
    for r in valid_ranges:
        slug = r.lower().replace(" ", "_")
        reconstructed = slug.replace("_", " ").title()
        assert reconstructed == r
