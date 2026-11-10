"""Reliability, offline fallback, caching, and graceful degradation tests for Nivara."""
import time
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import pytest
import requests

from campus_energy.dashboard.agent import (
    chat_with_gemini,
    generate_offline_fallback,
    parse_actions_from_response,
    actions_to_recommendations,
)
from campus_energy.dashboard.app import (
    create_2d_campus_map_figure,
    cached_load_data,
    cached_engineer_features,
    cached_detect_anomalies,
    cached_generate_24h_forecast,
)
from campus_energy.profiles import load_facility_profile

DATA_DIR = Path(__file__).parent.parent / "data" / "generated"


@pytest.fixture
def sample_dfs():
    if not (DATA_DIR / "assets.csv").exists():
        pytest.skip("Generated dataset not found.")
    assets = pd.read_csv(DATA_DIR / "assets.csv")
    events = pd.read_csv(DATA_DIR / "events.csv")
    complaints = (
        pd.read_csv(DATA_DIR / "complaints.csv")
        if (DATA_DIR / "complaints.csv").exists()
        else pd.DataFrame()
    )
    return assets, events, complaints


def test_gemini_offline_fallback_when_api_key_none(sample_dfs):
    """Test Gemini offline fallback returns structured response with recommendations when API key is None."""
    assets, events, complaints = sample_dfs
    profile = load_facility_profile("config/facility_profiles/university.yaml")

    # API key is None
    response = chat_with_gemini(
        api_key=None,
        messages=[{"role": "user", "content": "What operational actions do you recommend?"}],
        events_df=events,
        assets_df=assets,
        complaints_df=complaints,
        active_role="operations",
        profile=profile,
    )

    assert isinstance(response, str)
    assert len(response) > 50
    assert "Offline Facility Intelligence Mode" in response
    assert "Error connecting to Gemini API" not in response

    # Parse actions from offline response
    actions = parse_actions_from_response(response)
    assert len(actions) >= 1
    assert any(a["action_type"] in ["Equipment Shutdown", "HVAC Inspection/Schedule", "Schedule Optimisation"] for a in actions)

    # Convert actions to defensible recommendations
    recs = actions_to_recommendations(actions, profile)
    assert len(recs) >= 1
    for r in recs:
        assert r.savings.inr.expected > 0
        assert r.savings.kwh.expected > 0
        assert r.facility_id == profile.facility_id


def test_gemini_offline_fallback_when_api_key_empty(sample_dfs):
    """Test Gemini offline fallback works when API key is empty string or whitespace."""
    assets, events, complaints = sample_dfs
    response = chat_with_gemini(
        api_key="   ",
        messages=[{"role": "user", "content": "Status overview"}],
        events_df=events,
        assets_df=assets,
        complaints_df=complaints,
        active_role="administrator",
    )
    assert isinstance(response, str)
    assert "Offline Facility Intelligence Mode" in response


@pytest.mark.parametrize("query,expected_keywords", [
    (
        "What are the critical anomalies right now?",
        ["Critical Facility Anomaly Report", "Recommendation", "HVAC", "Equipment Shutdown"]
    ),
    (
        "How can we maximize savings and financial ROI?",
        ["Campus Energy Savings", "Economic Baseline Tariff", "IPMVP Option B", "Schedule Optimisation"]
    ),
    (
        "How do we manage peak demand and avoid ToD tariff penalties?",
        ["Peak Demand", "18:00 – 22:00", "Pre-Cooling", "Renewable Integration Review"]
    ),
])
def test_offline_fallback_specific_queries(sample_dfs, query, expected_keywords):
    """Test offline fallback handles specific queries ('critical anomalies', 'savings', 'peak demand') cleanly."""
    assets, events, complaints = sample_dfs
    profile = load_facility_profile("config/facility_profiles/university.yaml")

    response = generate_offline_fallback(
        messages=[{"role": "user", "content": query}],
        events_df=events,
        assets_df=assets,
        complaints_df=complaints,
        active_role="operations",
        profile=profile,
    )

    for kw in expected_keywords:
        assert kw.lower() in response.lower(), f"Expected keyword '{kw}' not found in response:\n{response}"

    # Verify structured actions can be parsed
    actions = parse_actions_from_response(response)
    assert len(actions) >= 1, f"No actions parsed from query '{query}'"


def test_offline_fallback_on_network_failure(sample_dfs):
    """Test offline fallback seamlessly catches network unreachable errors without raising."""
    assets, events, complaints = sample_dfs

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Network is unreachable")):
        response = chat_with_gemini(
            api_key="AIzaFakeKeyForOfflineTest",
            messages=[{"role": "user", "content": "What are the latest alerts?"}],
            events_df=events,
            assets_df=assets,
            complaints_df=complaints,
            active_role="operations",
        )

        assert isinstance(response, str)
        assert "Offline Facility Intelligence Mode" in response
        assert "Gemini API unreachable" in response
        assert "Traceback (most recent call last)" not in response
        actions = parse_actions_from_response(response)
        assert len(actions) >= 1


def test_map_fallback_data_generation():
    """Test 2D Plotly map fallback generation handles normal, alert, comfort, and empty modes."""
    # 1. Empty map data
    empty_fig = create_2d_campus_map_figure(df_map=pd.DataFrame())
    assert empty_fig is not None
    assert len(empty_fig.data) >= 1  # Should still have campus boundary / buildings

    # 2. Power telemetry mode with normal + alert markers
    sample_power_data = pd.DataFrame([
        {
            "id": "BLD-A-MTR",
            "name": "Building BLD-A Main Incomer",
            "category": "Building Main Meters",
            "building": "BLD-A",
            "lon": 84.8185,
            "lat": 22.2480,
            "power_kw": 45.2,
            "peak_kw": 62.0,
            "day_kwh": 480.0,
            "status": "ACTIVE",
            "color_hex": "#10b981",
            "alert": False,
        },
        {
            "id": "BLD-B-HVAC",
            "name": "Building BLD-B HVAC Chiller",
            "category": "HVAC / Climate",
            "building": "BLD-B",
            "lon": 84.8195,
            "lat": 22.2485,
            "power_kw": 88.5,
            "peak_kw": 95.0,
            "day_kwh": 920.0,
            "status": "ACTIVE",
            "color_hex": "#ef4444",
            "alert": True,
            "alert_type": "Baseline Exceedance",
            "alert_desc": "Continuous baseload anomaly during off-hours",
        },
    ])

    power_fig = create_2d_campus_map_figure(df_map=sample_power_data, is_comfort_mode=False)
    assert power_fig is not None
    trace_names = [t.name for t in power_fig.data if t.name]
    assert any("Active Anomalies" in n for n in trace_names)
    assert any("Normal" in n for n in trace_names)

    # 3. Comfort mode with PMV / PPD categories
    sample_comfort_data = pd.DataFrame([
        {
            "id": "ZONE-1",
            "name": "Classroom 101",
            "category": "Comfortable (Neutral)",
            "building": "BLD-A",
            "zone": "CR101",
            "lon": 84.8182,
            "lat": 22.2478,
            "temp_c": "24.2 °C",
            "humidity": "52.0%",
            "pmv": "+0.15",
            "ppd": 5.4,
            "advisory": "Maintain optimal setpoint",
            "color_hex": "#10b981",
        },
        {
            "id": "ZONE-2",
            "name": "Computer Lab 1",
            "category": "Warm / Heat Discomfort",
            "building": "BLD-B",
            "zone": "LAB1",
            "lon": 84.8198,
            "lat": 22.2482,
            "temp_c": "28.5 °C",
            "humidity": "68.0%",
            "pmv": "+1.85",
            "ppd": 69.2,
            "advisory": "Increase AC cooling capacity",
            "color_hex": "#ef4444",
        },
    ])

    comfort_fig = create_2d_campus_map_figure(df_map=sample_comfort_data, is_comfort_mode=True)
    assert comfort_fig is not None
    c_trace_names = [t.name for t in comfort_fig.data if t.name]
    assert "Comfortable (Neutral)" in c_trace_names
    assert "Warm / Heat Discomfort" in c_trace_names


def test_caching_idempotency_and_speed():
    """Test caching idempotency and fast repeated execution for rapid demo day page switching."""
    if not (DATA_DIR / "telemetry.csv").exists():
        pytest.skip("Generated dataset not found.")

    # 1. Test cached_load_data idempotency
    d1 = cached_load_data(str(DATA_DIR))
    d2 = cached_load_data(str(DATA_DIR))
    assert d1 is not None and d2 is not None
    assert len(d1[0]) == len(d2[0])
    assert len(d1[1]) == len(d2[1])

    assets, tel, ctx, events, complaints = d1

    # 2. Test cached_engineer_features
    # Warmup / Call 1
    t0 = time.perf_counter()
    feat1 = cached_engineer_features(tel, ctx)
    t_first = time.perf_counter() - t0

    # Call 2 (Cache hit)
    t1 = time.perf_counter()
    feat2 = cached_engineer_features(tel, ctx)
    t_second = time.perf_counter() - t1

    assert feat1.equals(feat2)
    assert t_second < 1.0, f"Cache hit took {t_second:.4f}s, expected < 1.0s"
    assert t_second < t_first

    # 3. Test cached_detect_anomalies
    alerts1 = cached_detect_anomalies(feat1, assets)
    alerts2 = cached_detect_anomalies(feat1, assets)
    assert len(alerts1) == len(alerts2)
    assert [a.asset_id for a in alerts1] == [a.asset_id for a in alerts2]

    # 4. Test cached_generate_24h_forecast
    fc1 = cached_generate_24h_forecast(tel, ctx, assets, temp_offset=0.0)
    fc2 = cached_generate_24h_forecast(tel, ctx, assets, temp_offset=0.0)
    assert fc1.equals(fc2)
