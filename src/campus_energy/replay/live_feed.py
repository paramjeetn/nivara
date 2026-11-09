"""Live telemetry feed module for real-time streaming simulation.

Provides helper utilities to extract sliding windows of telemetry readings
for real-time dashboard tickers and live MQTT / IoT ingestion simulations.
"""
from __future__ import annotations

import logging
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)


def get_live_telemetry_window(
    telemetry_df: pd.DataFrame | None,
    step_index: int = 0,
    window_size: int = 5,
) -> list[dict[str, Any]]:
    """Get a sliding window of telemetry readings to simulate real-time ingestion.

    Args:
        telemetry_df: DataFrame containing telemetry readings.
        step_index: Current step / tick offset in the replay timeline.
        window_size: Number of consecutive telemetry records to retrieve.

    Returns:
        List of formatted reading dictionaries.
    """
    if telemetry_df is None or telemetry_df.empty:
        return []

    # Sort by timestamp if column exists and not already sorted
    if "timestamp" in telemetry_df.columns:
        if not telemetry_df["timestamp"].is_monotonic_increasing:
            df_sorted = telemetry_df.sort_values("timestamp").reset_index(drop=True)
        else:
            df_sorted = telemetry_df.reset_index(drop=True)
    else:
        df_sorted = telemetry_df.reset_index(drop=True)

    total_rows = len(df_sorted)
    if total_rows == 0:
        return []

    window_size = max(1, window_size)
    actual_count = min(window_size, total_rows)

    start_offset = step_index % total_rows
    indices = [(start_offset + i) % total_rows for i in range(actual_count)]
    slice_df = df_sorted.iloc[indices]

    records: list[dict[str, Any]] = []
    for idx, row in slice_df.iterrows():
        reading_id = str(row["reading_id"]) if "reading_id" in row and pd.notna(row["reading_id"]) else f"R-{idx:08d}"
        timestamp_val = str(row["timestamp"]) if "timestamp" in row and pd.notna(row["timestamp"]) else ""
        asset_id = str(row["asset_id"]) if "asset_id" in row and pd.notna(row["asset_id"]) else "UNKNOWN"
        reading_type = str(row.get("reading_type", "equipment")) if pd.notna(row.get("reading_type")) else "equipment"

        power_val = row.get("power_kw")
        power_kw = round(float(power_val), 2) if pd.notna(power_val) else 0.0

        voltage_val = row.get("voltage_v")
        voltage_v = round(float(voltage_val), 1) if pd.notna(voltage_val) else 230.0

        pf_val = row.get("power_factor")
        power_factor = round(float(pf_val), 2) if pd.notna(pf_val) else 0.92

        quality_flag = str(row.get("quality_flag", "valid")) if pd.notna(row.get("quality_flag")) else "valid"
        sensor_status = str(row.get("sensor_status", "healthy")) if pd.notna(row.get("sensor_status")) else "healthy"

        records.append({
            "reading_id": reading_id,
            "timestamp": timestamp_val,
            "asset_id": asset_id,
            "reading_type": reading_type,
            "power_kw": power_kw,
            "voltage_v": voltage_v,
            "power_factor": power_factor,
            "quality_flag": quality_flag,
            "sensor_status": sensor_status,
        })

    return records


def get_telemetry_feed_stats(telemetry_df: pd.DataFrame | None) -> dict[str, Any]:
    """Calculate summary statistics for the telemetry feed.

    Args:
        telemetry_df: DataFrame of telemetry records.

    Returns:
        Dictionary with total_readings, latest_timestamp, avg_power_kw, active_sensors.
    """
    if telemetry_df is None or telemetry_df.empty:
        return {
            "total_readings": 0,
            "latest_timestamp": "",
            "avg_power_kw": 0.0,
            "active_sensors": 0,
        }

    total_readings = int(len(telemetry_df))

    latest_ts = ""
    if "timestamp" in telemetry_df.columns:
        max_ts = telemetry_df["timestamp"].max()
        latest_ts = str(max_ts) if pd.notna(max_ts) else ""

    avg_power_kw = 0.0
    if "power_kw" in telemetry_df.columns:
        valid_power = telemetry_df["power_kw"].dropna()
        if not valid_power.empty:
            avg_power_kw = round(float(valid_power.mean()), 2)

    active_sensors = 0
    if "asset_id" in telemetry_df.columns:
        active_sensors = int(telemetry_df["asset_id"].nunique())

    return {
        "total_readings": total_readings,
        "latest_timestamp": latest_ts,
        "avg_power_kw": avg_power_kw,
        "active_sensors": active_sensors,
    }
