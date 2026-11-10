"""Unit and integration tests for Assets Destination View (Step UX-6).
=====================================================================
Tests:
1. Asset list data assembly & enrichment (current kW, expected kW, deltas, status).
2. Filter bar operations (by building, type, protection status, search query).
3. Asset detail view data preparation for normal equipment.
4. Asset detail view data preparation for protected mission-critical equipment.
5. Operational history aggregation (events + work orders with status and SLA).
6. 2D Map fallback vector schematic generation for power and comfort overlays.
7. Query parameter deep-linking (?page=assets&asset=EQ-UTL-P1-PM) and back routing.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from campus_energy.decision.ledger import SavingsLedger, WorkOrderItem
from campus_energy.decision.recommendation import Recommendation, SavingsBundle
from campus_energy.profiles import FacilityProfile, ProtectedLoad, load_all_profiles
from campus_energy.ui.views.assets import (
    create_2d_campus_map_figure,
    filter_assets,
    prepare_asset_detail_data,
    prepare_asset_list_data,
    render_assets,
)


@pytest.fixture
def sample_facility_profile() -> FacilityProfile:
    """Load sample facility profile with protected loads."""
    profiles = load_all_profiles("config/facility_profiles")
    profile = profiles.get("university") or list(profiles.values())[0]
    return profile


@pytest.fixture
def sample_assets_df() -> pd.DataFrame:
    """Fixture with a variety of equipment and meters across buildings."""
    return pd.DataFrame([
        {
            "asset_id": "EQ-UTL-P1-PM",
            "asset_name": "Primary Water Pump 1",
            "asset_type": "pump",
            "building_id": "BLD-UTL",
            "zone_id": "BLD-UTL-P1",
            "rated_power_kw": 7.5,
            "equipment_count": 1,
            "monitoring_level": "equipment",
        },
        {
            "asset_id": "EQ-UTL-P2-PM",
            "asset_name": "Secondary Water Pump 2",
            "asset_type": "pump",
            "building_id": "BLD-UTL",
            "zone_id": "BLD-UTL-P2",
            "rated_power_kw": 7.5,
            "equipment_count": 1,
            "monitoring_level": "equipment",
        },
        {
            "asset_id": "EQ-ADM-OF101-AC",
            "asset_name": "Administration Block AC Unit",
            "asset_type": "hvac",
            "building_id": "BLD-ADM",
            "zone_id": "BLD-ADM-OF101",
            "rated_power_kw": 12.0,
            "equipment_count": 1,
            "monitoring_level": "equipment",
        },
        {
            "asset_id": "EQ-A-CR101-LT",
            "asset_name": "Classroom 101 Lights",
            "asset_type": "lighting",
            "building_id": "BLD-A",
            "zone_id": "BLD-A-CR101",
            "rated_power_kw": 1.8,
            "equipment_count": 20,
            "monitoring_level": "equipment",
        },
        {
            "asset_id": "MTR-BLD-A",
            "asset_name": "Academic Block A Main Incomer",
            "asset_type": "meter",
            "building_id": "BLD-A",
            "zone_id": "—",
            "rated_power_kw": 150.0,
            "equipment_count": 1,
            "monitoring_level": "building",
        },
    ])


@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    """Generate 24 hours of 5-minute interval telemetry for the test assets."""
    base_time = datetime(2026, 11, 10, 0, 0, tzinfo=timezone.utc)
    rows = []
    # 288 5-min intervals in 24 hours
    for step in range(288):
        ts = base_time + timedelta(minutes=5 * step)
        # Pump 1: active draw
        rows.append({
            "timestamp": ts,
            "asset_id": "EQ-UTL-P1-PM",
            "reading_type": "equipment",
            "power_kw": 6.8 + (step % 5) * 0.1,
            "interval_energy_kwh": 0.55,
            "temperature_c": 32.0,
            "state": "ON",
        })
        # Pump 2: sibling peer
        rows.append({
            "timestamp": ts,
            "asset_id": "EQ-UTL-P2-PM",
            "reading_type": "equipment",
            "power_kw": 5.2 + (step % 3) * 0.1,
            "interval_energy_kwh": 0.43,
            "temperature_c": 30.5,
            "state": "ON",
        })
        # HVAC
        rows.append({
            "timestamp": ts,
            "asset_id": "EQ-ADM-OF101-AC",
            "reading_type": "equipment",
            "power_kw": 10.5 if (step >= 96 and step <= 216) else 0.8,
            "interval_energy_kwh": 0.85 if (step >= 96 and step <= 216) else 0.07,
            "temperature_c": 23.5,
            "state": "ON" if (step >= 96 and step <= 216) else "OFF",
        })
        # Lighting
        rows.append({
            "timestamp": ts,
            "asset_id": "EQ-A-CR101-LT",
            "reading_type": "equipment",
            "power_kw": 1.7 if (step >= 96 and step <= 216) else 0.0,
            "interval_energy_kwh": 0.14 if (step >= 96 and step <= 216) else 0.0,
            "temperature_c": 24.0,
            "state": "ON" if (step >= 96 and step <= 216) else "OFF",
        })
        # Meter
        rows.append({
            "timestamp": ts,
            "asset_id": "MTR-BLD-A",
            "reading_type": "meter",
            "power_kw": 48.0 + (step % 10),
            "interval_energy_kwh": 4.0,
            "temperature_c": 0.0,
            "state": "ACTIVE",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_events_df() -> pd.DataFrame:
    """Sample anomaly ground truth events."""
    return pd.DataFrame([
        {
            "event_id": "EVT-CEF449F8",
            "asset_id": "EQ-UTL-P1-PM",
            "event_type": "pump_no_flow",
            "start_time": "2026-11-10T11:30:00+05:30",
            "end_time": "2026-11-10T13:00:00+05:30",
            "severity": "critical",
            "description": "Pump running dry with zero flow sensor indication",
        },
        {
            "event_id": "EVT-54D71E1B",
            "asset_id": "EQ-ADM-OF101-AC",
            "event_type": "hvac_left_on",
            "start_time": "2026-11-10T19:00:00+05:30",
            "end_time": "2026-11-10T23:00:00+05:30",
            "severity": "medium",
            "description": "After-hours HVAC continuous run",
        },
    ])


@pytest.fixture
def sample_ledger() -> SavingsLedger:
    """Sample SavingsLedger instance with work orders targeting assets."""
    rec = Recommendation(
        id="REC-PUMP-01",
        facility_id="university",
        type="anomaly_fix",
        target="EQ-UTL-P1-PM",
        action_text="Inspect discharge isolation valve and prime impeller casing",
        trigger="pump_no_flow",
        savings=SavingsBundle.from_expected(kwh_expected=240.0),
    )
    ledger = SavingsLedger()
    ledger.add_recommendation(rec)
    wo = ledger.accept_recommendation(rec.id)
    return ledger


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Asset List Data Assembly & Enrichment
# ─────────────────────────────────────────────────────────────────────────────

def test_prepare_asset_list_data(
    sample_facility_profile: FacilityProfile,
    sample_assets_df: pd.DataFrame,
    sample_telemetry_df: pd.DataFrame,
    sample_events_df: pd.DataFrame,
):
    """Test data assembly for asset list view."""
    prepared = prepare_asset_list_data(
        facility_meta=sample_facility_profile,
        assets_df=sample_assets_df,
        tel_df=sample_telemetry_df,
        events_df=sample_events_df,
    )

    assert not prepared.empty
    assert len(prepared) == len(sample_assets_df)

    # Required columns for the table
    expected_cols = [
        "asset_id", "asset_name", "asset_type", "building_id", "zone_id",
        "current_kw", "expected_kw", "delta_kw", "delta_pct",
        "status", "is_protected", "rated_power_kw", "relative_last_seen"
    ]
    for col in expected_cols:
        assert col in prepared.columns, f"Missing column: {col}"

    # Verify pump has critical status from critical event
    pump_row = prepared[prepared["asset_id"] == "EQ-UTL-P1-PM"].iloc[0]
    assert pump_row["status"] == "Critical"
    assert pump_row["current_kw"] > 0.0
    assert "relative_last_seen" in pump_row


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Filter Bar Operations
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_assets_operations(
    sample_facility_profile: FacilityProfile,
    sample_assets_df: pd.DataFrame,
    sample_telemetry_df: pd.DataFrame,
    sample_events_df: pd.DataFrame,
):
    """Test filtering by building, type, search string, and protection."""
    prepared = prepare_asset_list_data(
        facility_meta=sample_facility_profile,
        assets_df=sample_assets_df,
        tel_df=sample_telemetry_df,
        events_df=sample_events_df,
    )

    # 1. Building filter
    filtered_utl = filter_assets(prepared, building="BLD-UTL")
    assert len(filtered_utl) == 2
    assert all(filtered_utl["building_id"] == "BLD-UTL")

    # 2. Asset type filter
    filtered_pumps = filter_assets(prepared, asset_type="pump")
    assert len(filtered_pumps) == 2
    assert all(filtered_pumps["asset_type"] == "pump")

    # 3. Search query
    filtered_search = filter_assets(prepared, search_query="Administration")
    assert len(filtered_search) == 1
    assert filtered_search.iloc[0]["asset_id"] == "EQ-ADM-OF101-AC"

    # 4. Status filter
    filtered_crit = filter_assets(prepared, status="Critical")
    assert len(filtered_crit) >= 1
    assert any(filtered_crit["asset_id"] == "EQ-UTL-P1-PM")

    # 5. Combined filter with no matches
    empty_res = filter_assets(prepared, building="BLD-UTL", asset_type="lighting")
    assert empty_res.empty


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Asset Detail Data Preparation for Normal Equipment
# ─────────────────────────────────────────────────────────────────────────────

def test_prepare_asset_detail_data_normal(
    sample_facility_profile: FacilityProfile,
    sample_assets_df: pd.DataFrame,
    sample_telemetry_df: pd.DataFrame,
    sample_events_df: pd.DataFrame,
    sample_ledger: SavingsLedger,
):
    """Test data assembly for an asset detail page."""
    detail = prepare_asset_detail_data(
        asset_id="EQ-ADM-OF101-AC",
        facility_meta=sample_facility_profile,
        assets_df=sample_assets_df,
        tel_df=sample_telemetry_df,
        events_df=sample_events_df,
        ledger=sample_ledger,
    )

    assert detail is not None
    assert detail["asset_row"]["asset_id"] == "EQ-ADM-OF101-AC"
    assert detail["mean_kw"] > 0
    assert detail["peak_kw"] >= detail["mean_kw"]
    assert detail["total_kwh"] > 0

    # 3h rolling mean
    assert "rolling_3h_kw" in detail["telemetry"].columns

    # Heatmap matrix
    hm = detail["heatmap_pivot"]
    assert not hm.empty
    assert hm.shape[1] <= 24  # 24 hours


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Protected Mission-Critical Load Handling
# ─────────────────────────────────────────────────────────────────────────────

def test_prepare_asset_detail_data_protected(
    sample_assets_df: pd.DataFrame,
    sample_telemetry_df: pd.DataFrame,
):
    """Test protected load metadata extraction and non-curtailability rationale."""
    # Create profile with EQ-UTL-P1-PM as protected
    protected_load = ProtectedLoad(
        asset_id="EQ-UTL-P1-PM",
        asset_name="Primary Water Pump 1",
        category="life_safety",
        curtailable=False,
        min_power_kw=4.0,
        reason="Fire hydrants & campus emergency water pressurization system",
    )
    profile = FacilityProfile(
        facility_id="test_fac",
        facility_name="Test Engineering Facility",
        facility_type="university",
        location="Odisha",
        floor_area_m2=10000.0,
        peak_occupancy=2000,
        data_dir="data/generated",
        tariff_profile="odisha_ht_commercial",
        protected_loads=[protected_load],
    )

    detail = prepare_asset_detail_data(
        asset_id="EQ-UTL-P1-PM",
        facility_meta=profile,
        assets_df=sample_assets_df,
        tel_df=sample_telemetry_df,
    )

    assert detail["is_protected"] is True
    assert detail["prot_category"] == "life_safety"
    assert "Fire hydrants" in detail["prot_reason"]
    assert detail["min_power_kw"] == 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Peer Comparison Generation
# ─────────────────────────────────────────────────────────────────────────────

def test_peer_comparison_generation(
    sample_facility_profile: FacilityProfile,
    sample_assets_df: pd.DataFrame,
    sample_telemetry_df: pd.DataFrame,
):
    """Test peer benchmarking against siblings of the same type in the same building."""
    detail = prepare_asset_detail_data(
        asset_id="EQ-UTL-P1-PM",
        facility_meta=sample_facility_profile,
        assets_df=sample_assets_df,
        tel_df=sample_telemetry_df,
    )

    peer_stats = detail["peer_stats"]
    assert not peer_stats.empty
    assert len(peer_stats) == 2  # P1 and P2
    assert "EQ-UTL-P1-PM" in peer_stats["asset_id"].values
    assert "EQ-UTL-P2-PM" in peer_stats["asset_id"].values

    selected_row = peer_stats[peer_stats["asset_id"] == "EQ-UTL-P1-PM"].iloc[0]
    assert bool(selected_row["is_selected"]) is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Operational Audit History Assembly
# ─────────────────────────────────────────────────────────────────────────────

def test_operational_history_assembly(
    sample_facility_profile: FacilityProfile,
    sample_assets_df: pd.DataFrame,
    sample_telemetry_df: pd.DataFrame,
    sample_events_df: pd.DataFrame,
    sample_ledger: SavingsLedger,
):
    """Test aggregation of anomaly alerts and maintenance work orders for an asset."""
    detail = prepare_asset_detail_data(
        asset_id="EQ-UTL-P1-PM",
        facility_meta=sample_facility_profile,
        assets_df=sample_assets_df,
        tel_df=sample_telemetry_df,
        events_df=sample_events_df,
        ledger=sample_ledger,
    )

    history = detail["audit_history"]
    assert len(history) >= 2  # Anomaly event + Work order

    types = [h["record_type"] for h in history]
    assert "Anomaly Alert" in types
    assert "Work Order" in types

    wo_record = [h for h in history if h["record_type"] == "Work Order"][0]
    assert "REC-PUMP-01" in wo_record["id"] or "WO-" in wo_record["id"]
    assert "SLA" in wo_record["sla_impact"]


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: 2D Campus Map Schematic Fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_map_fallback_generation():
    """Test that 2D schematic generates valid figures for empty, power, and comfort data."""
    # 1. Empty data
    fig_empty = create_2d_campus_map_figure(df_map=pd.DataFrame())
    assert fig_empty is not None
    assert len(fig_empty.data) >= 1  # Boundary & building polygons

    # 2. Power data
    power_df = pd.DataFrame([
        {
            "id": "EQ-UTL-P1-PM",
            "name": "Water Pump 1",
            "category": "Water Pumps",
            "building": "BLD-UTL",
            "lon": 84.8210,
            "lat": 22.2465,
            "power_kw": 7.2,
            "peak_kw": 8.0,
            "day_kwh": 85.0,
            "status": "ACTIVE",
            "color_hex": "#ef4444",
            "alert": True,
            "alert_type": "Pump Cavitation",
            "alert_desc": "Excess vibration and drop in flow",
        }
    ])
    fig_power = create_2d_campus_map_figure(df_map=power_df, is_comfort_mode=False)
    assert fig_power is not None
    names = [t.name for t in fig_power.data if t.name]
    assert any("Active Anomalies" in n for n in names)

    # 3. Comfort data
    comfort_df = pd.DataFrame([
        {
            "id": "BLD-A-CR101",
            "name": "Classroom 101",
            "category": "Comfortable (Neutral)",
            "building": "BLD-A",
            "zone": "CR101",
            "lon": 84.8185,
            "lat": 22.2480,
            "temp_c": "24.2 °C",
            "humidity": "52.0%",
            "pmv": "+0.12",
            "ppd": 5.2,
            "advisory": "Nominal",
            "color_hex": "#10b981",
        }
    ])
    fig_comfort = create_2d_campus_map_figure(df_map=comfort_df, is_comfort_mode=True)
    assert fig_comfort is not None
    c_names = [t.name for t in fig_comfort.data if t.name]
    assert "Comfortable (Neutral)" in c_names


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Query Parameter Deep-Linking (?page=assets&asset=...)
# ─────────────────────────────────────────────────────────────────────────────

def test_query_param_deep_linking(
    sample_facility_profile: FacilityProfile,
    sample_assets_df: pd.DataFrame,
    sample_telemetry_df: pd.DataFrame,
    sample_events_df: pd.DataFrame,
):
    """Test that specifying asset ID in query params routes directly to detail page."""
    with patch("streamlit.query_params", {"page": "assets", "asset": "EQ-UTL-P1-PM"}), \
         patch("streamlit.session_state", {}), \
         patch("campus_energy.ui.views.assets.render_asset_detail_page") as mock_detail:

        render_assets(
            facility_meta=sample_facility_profile,
            assets=sample_assets_df,
            tel=sample_telemetry_df,
            ctx=pd.DataFrame(),
            events=sample_events_df,
        )

        mock_detail.assert_called_once()
        call_kwargs = mock_detail.call_args[1]
        assert call_kwargs["asset_id"] == "EQ-UTL-P1-PM"
