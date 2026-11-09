"""Unit tests for live telemetry feed module."""
import pandas as pd
import pytest
from campus_energy.replay.live_feed import (
    get_live_telemetry_window,
    get_telemetry_feed_stats,
)


@pytest.fixture
def sample_telemetry_df() -> pd.DataFrame:
    """Create a sample telemetry DataFrame."""
    return pd.DataFrame({
        "reading_id": [f"R-{i:03d}" for i in range(1, 11)],
        "timestamp": [
            f"2025-09-01T{i:02d}:00:00+05:30" for i in range(10)
        ],
        "asset_id": [f"MTR-00{i % 3 + 1}" for i in range(10)],
        "reading_type": ["meter" if i % 2 == 0 else "equipment" for i in range(10)],
        "power_kw": [10.5 + i * 2.0 for i in range(10)],
        "voltage_v": [230.0 + (i % 5) for i in range(10)],
        "power_factor": [0.90 + (i % 8) * 0.01 for i in range(10)],
        "quality_flag": ["valid"] * 9 + ["flagged"],
        "sensor_status": ["healthy"] * 8 + ["degraded", "faulty"],
    })


def test_empty_dataframe():
    """Verify empty or None DataFrames return empty window and zero stats."""
    assert get_live_telemetry_window(None) == []
    assert get_live_telemetry_window(pd.DataFrame()) == []

    empty_stats = get_telemetry_feed_stats(None)
    assert empty_stats["total_readings"] == 0
    assert empty_stats["latest_timestamp"] == ""
    assert empty_stats["avg_power_kw"] == 0.0
    assert empty_stats["active_sensors"] == 0

    df_empty_stats = get_telemetry_feed_stats(pd.DataFrame())
    assert df_empty_stats["total_readings"] == 0


def test_live_telemetry_window_structure(sample_telemetry_df):
    """Verify the keys and format of returned telemetry window readings."""
    window = get_live_telemetry_window(sample_telemetry_df, step_index=0, window_size=5)
    assert len(window) == 5

    first = window[0]
    expected_keys = {
        "reading_id",
        "timestamp",
        "asset_id",
        "reading_type",
        "power_kw",
        "voltage_v",
        "power_factor",
        "quality_flag",
        "sensor_status",
    }
    assert expected_keys.issubset(first.keys())
    assert first["reading_id"] == "R-001"
    assert first["asset_id"] == "MTR-001"
    assert first["power_kw"] == 10.5
    assert first["voltage_v"] == 230.0
    assert first["power_factor"] == 0.90
    assert first["quality_flag"] == "valid"
    assert first["sensor_status"] == "healthy"


def test_step_indexing_and_modulo_wrapping(sample_telemetry_df):
    """Verify step indexing progresses and wraps around modulo total length."""
    # Step index 0 gives items 0..4
    w0 = get_live_telemetry_window(sample_telemetry_df, step_index=0, window_size=3)
    assert [r["reading_id"] for r in w0] == ["R-001", "R-002", "R-003"]

    # Step index 8 with window 4 wraps around (length is 10, so 8, 9, 0, 1)
    w8 = get_live_telemetry_window(sample_telemetry_df, step_index=8, window_size=4)
    assert [r["reading_id"] for r in w8] == ["R-009", "R-010", "R-001", "R-002"]

    # Step index 10 (exact multiple) is equivalent to step index 0
    w10 = get_live_telemetry_window(sample_telemetry_df, step_index=10, window_size=3)
    assert [r["reading_id"] for r in w10] == ["R-001", "R-002", "R-003"]

    # Large step index works properly
    w_large = get_live_telemetry_window(sample_telemetry_df, step_index=102, window_size=2)
    assert [r["reading_id"] for r in w_large] == ["R-003", "R-004"]


def test_sorting_by_timestamp():
    """Verify telemetry window is sorted chronologically."""
    df_unsorted = pd.DataFrame({
        "reading_id": ["R-B", "R-A", "R-C"],
        "timestamp": [
            "2025-09-01T12:00:00+05:30",
            "2025-09-01T08:00:00+05:30",
            "2025-09-01T16:00:00+05:30",
        ],
        "asset_id": ["A1", "A2", "A3"],
    })
    window = get_live_telemetry_window(df_unsorted, step_index=0, window_size=3)
    assert [r["reading_id"] for r in window] == ["R-A", "R-B", "R-C"]


def test_missing_and_nan_values():
    """Verify safe fallbacks when values are missing or NaN."""
    df_nans = pd.DataFrame({
        "reading_id": ["R-NAN"],
        "timestamp": ["2025-09-01T00:00:00"],
        "asset_id": ["AST-01"],
        "power_kw": [None],
        "voltage_v": [float("nan")],
        "power_factor": [None],
    })
    window = get_live_telemetry_window(df_nans, step_index=0, window_size=1)
    assert len(window) == 1
    assert window[0]["power_kw"] == 0.0
    assert window[0]["voltage_v"] == 230.0
    assert window[0]["power_factor"] == 0.92
    assert window[0]["reading_type"] == "equipment"
    assert window[0]["quality_flag"] == "valid"
    assert window[0]["sensor_status"] == "healthy"


def test_get_telemetry_feed_stats(sample_telemetry_df):
    """Verify calculation of feed summary statistics."""
    stats = get_telemetry_feed_stats(sample_telemetry_df)
    assert stats["total_readings"] == 10
    assert stats["latest_timestamp"] == "2025-09-01T09:00:00+05:30"
    assert stats["active_sensors"] == 3
    # Average of 10.5, 12.5, 14.5, 16.5, 18.5, 20.5, 22.5, 24.5, 26.5, 28.5 = 19.5
    assert stats["avg_power_kw"] == 19.5
