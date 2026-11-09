"""Unit tests for thermal comfort analytics (ASHRAE 55 PMV / PPD)."""

import pandas as pd
import pytest

from campus_energy.analytics.comfort import (
    calculate_discomfort_index,
    calculate_pmv_ppd,
    get_campus_comfort_summary,
    get_comfort_snapshot,
)


def test_calculate_pmv_ppd_nominal():
    """Nominal room conditions (24C, 50% RH) should be in neutral comfort zone."""
    res = calculate_pmv_ppd(temp_c=24.0, humidity_pct=50.0)
    assert abs(res["pmv"]) <= 0.5, f"Expected |PMV| <= 0.5, got {res['pmv']}"
    assert res["ppd"] <= 10.0, f"Expected PPD <= 10%, got {res['ppd']}%"
    assert res["category"] == "Comfortable (Neutral)"
    assert res["color"] == "#10b981"
    assert "optimal" in res["advisory"].lower()


def test_calculate_pmv_ppd_high_temp():
    """High temperature (36C) should yield heat discomfort with PMV > 1.5."""
    res = calculate_pmv_ppd(temp_c=36.0, humidity_pct=50.0)
    assert res["pmv"] > 1.5, f"Expected PMV > 1.5, got {res['pmv']}"
    assert res["category"] == "Warm / Heat Discomfort"
    assert res["color"] == "#ef4444"
    assert res["ppd"] > 50.0


def test_calculate_pmv_ppd_cold_temp():
    """Low temperature (16C) should yield cool / overcooled condition with PMV < -1.5."""
    res = calculate_pmv_ppd(temp_c=16.0, humidity_pct=50.0)
    assert res["pmv"] < -1.5, f"Expected PMV < -1.5, got {res['pmv']}"
    assert res["category"] == "Cool / Overcooled"
    assert res["color"] == "#6366f1"


def test_calculate_pmv_ppd_slightly_warm():
    """26.5C should be slightly warm with 0.5 < PMV <= 1.5."""
    res = calculate_pmv_ppd(temp_c=26.5, humidity_pct=50.0)
    assert 0.5 < res["pmv"] <= 1.5, f"Expected 0.5 < PMV <= 1.5, got {res['pmv']}"
    assert res["category"] == "Slightly Warm"
    assert res["color"] == "#f59e0b"


def test_calculate_discomfort_index():
    """Discomfort index should compute sensible Thom categories."""
    res = calculate_discomfort_index(temp_c=28.0, humidity_pct=75.0)
    assert "di" in res
    assert res["di"] > 24.0
    assert "Discomfort" in res["category"]


def test_get_comfort_snapshot_structure():
    """Snapshot should return dataframe with all expected columns."""
    ctx_data = {
        "timestamp": ["2025-09-01T12:00:00", "2025-09-01T12:00:00"],
        "building_id": ["BLD-A", "BLD-B"],
        "zone_id": ["BLD-A-CR101", "BLD-B-CR201"],
        "outdoor_temperature_c": [25.0, 26.0],
        "humidity_pct": [55.0, 50.0],
    }
    assets_data = {
        "asset_id": ["BLD-A-CR101", "BLD-B-CR201"],
        "asset_name": ["Classroom 101", "Classroom 201"],
        "asset_type": ["zone", "zone"],
        "building_id": ["BLD-A", "BLD-B"],
        "zone_id": ["BLD-A-CR101", "BLD-B-CR201"],
    }
    ctx_df = pd.DataFrame(ctx_data)
    assets_df = pd.DataFrame(assets_data)

    snapshot = get_comfort_snapshot(ctx_df, assets_df)

    expected_cols = [
        "building_id",
        "zone_id",
        "temperature_c",
        "humidity_pct",
        "pmv",
        "ppd",
        "category",
        "color",
        "advisory",
    ]
    for col in expected_cols:
        assert col in snapshot.columns, f"Missing column {col} in snapshot"

    assert len(snapshot) == 2
    assert snapshot.iloc[0]["zone_id"] == "BLD-A-CR101"
    assert snapshot.iloc[0]["category"] in ["Comfortable (Neutral)", "Slightly Warm"]


def test_get_campus_comfort_summary():
    """Summary should aggregate zone metrics correctly."""
    comfort_data = {
        "building_id": ["BLD-A", "BLD-B", "BLD-ADM"],
        "zone_id": ["BLD-A-CR101", "BLD-B-CR201", "BLD-ADM-OF101"],
        "temperature_c": [24.0, 24.5, 30.0],
        "humidity_pct": [50.0, 52.0, 60.0],
        "pmv": [0.0, 0.2, 1.8],
        "ppd": [5.0, 6.0, 68.0],
    }
    comfort_df = pd.DataFrame(comfort_data)
    summary = get_campus_comfort_summary(comfort_df)

    assert "avg_temp" in summary
    assert "avg_humidity" in summary
    assert "avg_pmv" in summary
    assert "avg_ppd" in summary
    assert "pct_compliant" in summary

    # Two out of three zones have |PMV| <= 0.5 (66.7%)
    assert 60.0 <= summary["pct_compliant"] <= 70.0
    assert summary["avg_temp"] == pytest.approx(26.2, rel=1e-1)
