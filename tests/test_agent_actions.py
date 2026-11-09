import pytest
from campus_energy.dashboard.agent import parse_actions_from_response


def test_parse_actions_from_response_sample():
    text = "You should shutdown the HVAC in Building A after hours to save 50 kWh"
    actions = parse_actions_from_response(text)
    assert len(actions) == 2
    types = [a["action_type"] for a in actions]
    assert "Equipment Shutdown" in types
    assert "HVAC Inspection/Schedule" in types
    for a in actions:
        assert a["target_asset"] == "Building A"
        assert a["suggested_time"] == "After 18:00"
        assert a["estimated_saving_kwh"] == 50.0
        assert a["status"] == "pending"
        assert "detected_at" in a


def test_parse_actions_empty_and_no_match():
    assert parse_actions_from_response("") == []
    assert parse_actions_from_response(None) == []
    assert parse_actions_from_response("All systems normal.") == []


def test_parse_actions_various_types_and_dedup():
    text = (
        "Urgent: Check the lighting and water pump in BLD-102 immediately. "
        "Also turn off lights if unoccupied."
    )
    actions = parse_actions_from_response(text)
    types = [a["action_type"] for a in actions]
    # "turn off" -> Equipment Shutdown
    # "lighting" / "lights" -> Lighting Audit (deduplicated)
    # "pump" / "water" -> Pump Maintenance (deduplicated)
    assert "Equipment Shutdown" in types
    assert "Lighting Audit" in types
    assert "Pump Maintenance" in types
    assert len(types) == len(set(types))  # No duplicates

    for a in actions:
        assert a["target_asset"] == "BLD-102"
        assert a["suggested_time"] == "Immediately"
        assert a["estimated_saving_kwh"] is None


def test_parse_actions_campus_wide_and_maintenance_window():
    text = "Please review the solar schedule to optimize grid export."
    actions = parse_actions_from_response(text)
    types = [a["action_type"] for a in actions]
    assert "Schedule Optimisation" in types
    assert "Renewable Integration Review" in types
    for a in actions:
        assert a["target_asset"] == "Campus-wide"
        assert a["suggested_time"] == "Next maintenance window"
        assert a["estimated_saving_kwh"] is None
