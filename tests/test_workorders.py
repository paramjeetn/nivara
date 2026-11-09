"""Unit tests for the Fault Priority Queue & Maintenance Work-Order View (TASK-09)."""
import pandas as pd
import pytest

from campus_energy.analytics.workorders import (
    COLUMNS,
    create_work_orders,
    determine_assigned_team,
    calculate_sla_status,
    calculate_priority_score,
    get_work_order_summary,
)


@pytest.fixture
def sample_assets():
    return pd.DataFrame([
        {
            "asset_id": "EQ-HVAC-01",
            "asset_type": "hvac",
            "asset_name": "Main Chiller Unit 1",
            "building_id": "BLD-ADMIN",
            "rated_power_kw": 25.0,
        },
        {
            "asset_id": "EQ-LT-101",
            "asset_type": "lighting",
            "asset_name": "Library Highbay Lights",
            "building_id": "BLD-LIB",
            "rated_power_kw": 3.5,
        },
        {
            "asset_id": "EQ-PUMP-02",
            "asset_type": "pump",
            "asset_name": "Rooftop Hydro Pump",
            "building_id": "BLD-UTL",
            "rated_power_kw": 7.5,
        },
        {
            "asset_id": "MTR-BLD-01",
            "asset_type": "meter",
            "asset_name": "Block 1 Smart Meter",
            "building_id": "BLD-ENG",
            "rated_power_kw": None,
        },
        {
            "asset_id": "EQ-GEN-99",
            "asset_type": "other",
            "asset_name": "Exhaust Fan 99",
            "building_id": "BLD-WORKSHOP",
            "rated_power_kw": 1.2,
        },
    ])


@pytest.fixture
def sample_events():
    return pd.DataFrame([
        {
            "event_id": "EVT-001",
            "asset_id": "EQ-HVAC-01",
            "event_type": "hvac_left_on",
            "severity": "critical",
            "start_time": "2025-09-01T18:00:00+05:30",
            "end_time": "2025-09-01T21:00:00+05:30",
            "description": "Chiller running at full load after hours",
        },
        {
            "event_id": "EVT-002",
            "asset_id": "EQ-LT-101",
            "event_type": "lights_left_on",
            "severity": "medium",
            "start_time": "2025-09-02T19:00:00+05:30",
            "end_time": "2025-09-02T23:00:00+05:30",
            "description": "Library lights left on",
        },
        {
            "event_id": "EVT-003",
            "asset_id": "EQ-PUMP-02",
            "event_type": "pump_no_flow",
            "severity": "high",
            "start_time": "2025-09-03T10:00:00+05:30",
            "end_time": "2025-09-03T11:30:00+05:30",
            "description": "Dry run on rooftop pump",
        },
        {
            "event_id": "EVT-004",
            "asset_id": "MTR-BLD-01",
            "event_type": "sensor_dropout",
            "severity": "low",
            "start_time": "2025-09-04T08:00:00+05:30",
            "end_time": "2025-09-04T09:00:00+05:30",
            "description": "Gateway dropped connection",
        },
    ])


def test_create_work_orders_columns_and_ordering(sample_events, sample_assets):
    wo_df = create_work_orders(sample_events, sample_assets)

    assert not wo_df.empty
    assert len(wo_df) == 4

    # Verify all expected columns exist
    for col in COLUMNS:
        assert col in wo_df.columns, f"Missing required column: {col}"

    # Verify descending ordering by priority_score
    scores = wo_df["priority_score"].tolist()
    assert scores == sorted(scores, reverse=True), "Work orders must be sorted descending by priority_score"

    # Verify work order IDs format
    for wo_id in wo_df["wo_id"]:
        assert wo_id.startswith("WO-")

    # Critical Chiller fault should be the top priority
    top_wo = wo_df.iloc[0]
    assert top_wo["asset_id"] == "EQ-HVAC-01"
    assert top_wo["severity"] == "critical"
    assert top_wo["assigned_team"] == "HVAC & Thermal Systems Team"


def test_create_work_orders_empty():
    # Empty DataFrame
    empty_df = create_work_orders(pd.DataFrame(), pd.DataFrame())
    assert isinstance(empty_df, pd.DataFrame)
    assert empty_df.empty
    assert list(empty_df.columns) == COLUMNS

    # None inputs
    none_df = create_work_orders(None, None)
    assert isinstance(none_df, pd.DataFrame)
    assert none_df.empty
    assert list(none_df.columns) == COLUMNS


def test_assigned_team_categorization():
    # HVAC
    assert determine_assigned_team(asset_type="hvac") == "HVAC & Thermal Systems Team"
    assert determine_assigned_team(event_type="hvac_left_on") == "HVAC & Thermal Systems Team"
    assert determine_assigned_team(asset_id="EQ-ADM-01-AC") == "HVAC & Thermal Systems Team"

    # Lighting
    assert determine_assigned_team(asset_type="lighting") == "Electrical & Lighting Maintenance"
    assert determine_assigned_team(event_type="lights_left_on") == "Electrical & Lighting Maintenance"
    assert determine_assigned_team(asset_id="EQ-CR-LT") == "Electrical & Lighting Maintenance"

    # Pump / Water
    assert determine_assigned_team(asset_type="pump") == "Plumbing & Hydraulics Team"
    assert determine_assigned_team(event_type="pump_no_flow") == "Plumbing & Hydraulics Team"
    assert determine_assigned_team(event_type="water_leak") == "Plumbing & Hydraulics Team"

    # BMS / IoT Automation
    assert determine_assigned_team(event_type="sensor_dropout") == "BMS & IoT Automation Team"
    assert determine_assigned_team(event_type="duplicate_reading") == "BMS & IoT Automation Team"
    assert determine_assigned_team(asset_type="meter") == "BMS & IoT Automation Team"

    # Other / General
    assert determine_assigned_team(asset_type="other", event_type="unspecified") == "General Facility Maintenance"


def test_sla_status_and_overdue():
    # Within SLA
    target, status = calculate_sla_status("critical", hours_unresolved=1.5)
    assert target == "4 hours"
    assert "Remaining" in status
    assert "P1" in status

    # Overdue SLA
    target, status = calculate_sla_status("critical", hours_unresolved=6.0)
    assert target == "4 hours"
    assert "🔴 Overdue by 2.0h" in status

    # High severity
    target_high, status_high = calculate_sla_status("high", hours_unresolved=4.0)
    assert target_high == "12 hours"
    assert "⚠️ P2 — 8.0h Remaining" in status_high


def test_priority_score_range():
    score_crit = calculate_priority_score("critical", est_waste_inr_day=1500.0, hours_unresolved=24.0)
    assert 90.0 <= score_crit <= 100.0

    score_low = calculate_priority_score("low", est_waste_inr_day=50.0, hours_unresolved=1.0)
    assert 0.0 <= score_low <= 30.0

    assert score_crit > score_low


def test_get_work_order_summary(sample_events, sample_assets):
    wo_df = create_work_orders(sample_events, sample_assets)
    summary = get_work_order_summary(wo_df)

    assert summary["total_count"] == 4
    assert summary["active_count"] == 4
    assert summary["critical_count"] >= 1
    assert summary["daily_waste_kwh"] > 0
    assert summary["daily_waste_inr"] > 0
    assert summary["monthly_recovery_inr"] == round(summary["daily_waste_inr"] * 30.0, 2)
    assert summary["top_priority_asset"] == "Main Chiller Unit 1"
    assert summary["top_priority_score"] > 0

    # Test summary with empty DataFrame
    empty_summary = get_work_order_summary(pd.DataFrame())
    assert empty_summary["active_count"] == 0
    assert empty_summary["daily_waste_kwh"] == 0.0
