"""Tests for BYOD CSV upload adapter, Data Quality Engine, and anomaly confidence calibration."""
import io
from datetime import datetime, timedelta
import pandas as pd
import pytest

from campus_energy.adapters.csv_upload import (
    DataIngestionError,
    detect_column_mapping,
    detect_unit_conversions,
    normalize_timezones,
    process_uploaded_csv,
)
from campus_energy.adapters.quality import (
    DataQualityEngine,
    evaluate_telemetry_quality,
)
from campus_energy.analytics.anomalies import calculate_confidence_score, detect_anomalies
from campus_energy.schemas import DataQualityReport, QualityFlag, SensorStatus, TelemetryReading


def test_third_party_csv_mapping_and_unit_conversion():
    """Verify non-standard headers ('RecordedAt', 'MeterName', 'ActivePower_W')

    are correctly mapped, Watts are converted to kW, and validated against TelemetryReading.
    """
    csv_data = """RecordedAt,MeterName,ActivePower_W,Volt,PF
2026-11-01 08:00:00,AHU_NORTH_01,15000,230.5,0.92
2026-11-01 08:15:00,AHU_NORTH_01,16200,229.8,0.91
2026-11-01 08:30:00,AHU_NORTH_01,17500,231.0,0.94
2026-11-01 08:45:00,AHU_NORTH_01,15800,230.0,0.93
"""
    df, report = process_uploaded_csv(csv_data)

    assert len(df) == 4
    assert "power_kw" in df.columns
    assert "timestamp" in df.columns
    assert "asset_id" in df.columns
    assert (df["asset_id"] == "AHU_NORTH_01").all()

    # Watts -> kW conversion check (15000 W -> 15.0 kW)
    assert pytest.approx(df["power_kw"].iloc[0], 0.01) == 15.0
    assert pytest.approx(df["power_kw"].iloc[1], 0.01) == 16.2
    assert pytest.approx(df["power_kw"].iloc[2], 0.01) == 17.5

    # Check TelemetryReading validation
    for _, row in df.iterrows():
        reading = TelemetryReading.model_validate(row.to_dict())
        assert reading.asset_id == "AHU_NORTH_01"
        assert reading.power_kw >= 0.0

    # Data Quality report
    assert report.total_records == 4
    assert report.quality_score == 100.0
    assert report.outlier_count == 0
    assert report.duplicate_count == 0
    assert report.stuck_sensor_count == 0
    assert report.missing_intervals_count == 0
    assert report.confidence_penalty == 0.0


def test_data_quality_missing_intervals():
    """Verify detection of timeline gaps / missing sampling intervals."""
    # Cadence is 15 minutes, but row 3 skips 45 minutes (missing 2 intervals)
    base_ts = datetime(2026, 11, 1, 9, 0, 0)
    timestamps = [
        base_ts,
        base_ts + timedelta(minutes=15),
        base_ts + timedelta(minutes=30),
        base_ts + timedelta(minutes=75),  # 45 min gap -> 2 missed 15-min intervals
        base_ts + timedelta(minutes=90),
    ]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "asset_id": ["MTR-01"] * 5,
        "power_kw": [10.0, 11.2, 10.8, 12.0, 11.5],
        "voltage_v": [230.0] * 5,
        "power_factor": [0.9] * 5,
    })

    tagged_df, report = evaluate_telemetry_quality(df, expected_interval_minutes=15)

    assert report.missing_intervals_count == 2
    assert report.total_records == 5
    assert report.quality_score < 100.0
    assert any("missing interval gaps" in r.lower() for r in report.recommendations)


def test_data_quality_duplicate_detection():
    """Verify duplicate readings (same asset + timestamp) are detected and tagged."""
    base_ts = datetime(2026, 11, 1, 10, 0, 0)
    df = pd.DataFrame({
        "timestamp": [base_ts, base_ts, base_ts + timedelta(minutes=15)],
        "asset_id": ["MTR-01", "MTR-01", "MTR-01"],
        "power_kw": [25.0, 25.0, 26.0],
    })

    tagged_df, report = evaluate_telemetry_quality(df)

    assert report.duplicate_count == 1
    assert tagged_df["quality_flag"].iloc[1] == QualityFlag.DUPLICATED.value
    assert tagged_df["sensor_status"].iloc[1] == SensorStatus.DEGRADED.value
    assert any("duplicate" in r.lower() for r in report.recommendations)


def test_data_quality_stuck_sensor():
    """Verify consecutive identical values (>= stuck threshold) are identified as frozen/stuck."""
    base_ts = datetime(2026, 11, 1, 0, 0, 0)
    # Generate 15 consecutive identical readings (threshold is 12)
    timestamps = [base_ts + timedelta(minutes=15 * i) for i in range(15)]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "asset_id": ["CHILLER-01"] * 15,
        "power_kw": [42.5] * 15,  # Exactly 42.5 kW with zero jitter
    })

    tagged_df, report = evaluate_telemetry_quality(df, stuck_threshold_steps=12)

    assert report.stuck_sensor_count == 15
    assert (tagged_df["quality_flag"] == QualityFlag.STALE.value).all()
    assert (tagged_df["sensor_status"] == SensorStatus.DEGRADED.value).all()
    assert any("stuck" in r.lower() for r in report.recommendations)


def test_data_quality_outliers():
    """Verify extreme physical outliers (negative power, PF > 1.0, voltage > 300V) are caught."""
    base_ts = datetime(2026, 11, 1, 12, 0, 0)
    df = pd.DataFrame({
        "timestamp": [
            base_ts,
            base_ts + timedelta(minutes=15),
            base_ts + timedelta(minutes=30),
            base_ts + timedelta(minutes=45),
        ],
        "asset_id": ["MTR-02"] * 4,
        "power_kw": [10.0, -5.2, 12.0, 11.0],         # -5.2 is negative power outlier
        "voltage_v": [230.0, 230.0, 345.0, 230.0],     # 345V exceeds 300V threshold
        "power_factor": [0.95, 0.90, 0.92, 1.25],      # 1.25 exceeds 1.0 limit
    })

    tagged_df, report = evaluate_telemetry_quality(df)

    assert report.outlier_count == 3
    # Rows 1, 2, 3 should be tagged as out_of_range
    assert tagged_df["quality_flag"].iloc[0] == QualityFlag.VALID.value
    assert tagged_df["quality_flag"].iloc[1] == QualityFlag.OUT_OF_RANGE.value
    assert tagged_df["quality_flag"].iloc[2] == QualityFlag.OUT_OF_RANGE.value
    assert tagged_df["quality_flag"].iloc[3] == QualityFlag.OUT_OF_RANGE.value
    assert any("outlier" in r.lower() for r in report.recommendations)


def test_malformed_csv_error_handling():
    """Verify graceful and descriptive error handling for invalid/malformed CSVs."""
    # 1. Empty file
    with pytest.raises(DataIngestionError, match="empty|no data"):
        process_uploaded_csv("")

    # 2. Missing power column
    no_power_csv = """Time,Device,Temperature
2026-11-01 10:00:00,DEV-01,24.5
2026-11-01 10:15:00,DEV-01,24.8
"""
    with pytest.raises(DataIngestionError, match="power column"):
        process_uploaded_csv(no_power_csv)

    # 3. Unparseable datetime strings
    bad_dates_csv = """Time,Power,Device
this_is_not_a_date,12.5,DEV-01
another_bad_date,13.0,DEV-01
"""
    with pytest.raises(DataIngestionError, match="Unparseable datetime strings"):
        process_uploaded_csv(bad_dates_csv)

    # 4. Non-numeric values in power column
    corrupted_pwr_csv = """Time,kW,Device
2026-11-01 10:00:00,corrupted_reading,DEV-01
2026-11-01 10:15:00,12.5,DEV-01
"""
    with pytest.raises(DataIngestionError, match="Non-numeric values found in power column"):
        process_uploaded_csv(corrupted_pwr_csv)


def test_anomaly_confidence_calibration():
    """Verify confidence penalty is applied to anomaly alerts when data quality is degraded."""
    # Base confidence calculation
    clean_conf = calculate_confidence_score(
        detector_type="rule",
        has_layer_agreement=False,
        deviation=10.0,
        baseline_power=50.0,
        rated_power=50.0,
        quality_penalty=0.0,
    )

    degraded_conf = calculate_confidence_score(
        detector_type="rule",
        has_layer_agreement=False,
        deviation=10.0,
        baseline_power=50.0,
        rated_power=50.0,
        quality_penalty=-0.20,
    )

    assert pytest.approx(clean_conf - 0.20, 0.01) == degraded_conf

    # Verify report confidence penalty tiers
    report_good = DataQualityReport(total_records=100, quality_score=96.0)
    assert report_good.confidence_penalty == 0.0

    report_fair = DataQualityReport(total_records=100, quality_score=85.0, confidence_penalty=-0.10)
    assert report_fair.confidence_penalty == -0.10

    report_poor = DataQualityReport(total_records=100, quality_score=68.0, confidence_penalty=-0.20)
    assert report_poor.confidence_penalty == -0.20


def test_custom_column_mapping_and_timezone_normalization():
    """Verify explicit column mapping override and timezone normalization."""
    raw_csv = """ts_str,meter_name,pwr
2026-11-01T10:00:00+05:30,LAB_01,8.4
2026-11-01T10:15:00+05:30,LAB_01,8.6
"""
    custom_map = {"ts_str": "timestamp", "meter_name": "asset_id", "pwr": "power_kw"}
    df, report = process_uploaded_csv(raw_csv, column_mapping=custom_map, target_tz="UTC")

    assert len(df) == 2
    assert df["asset_id"].iloc[0] == "LAB_01"
    assert pytest.approx(df["power_kw"].iloc[0], 0.01) == 8.4
    # Check UTC conversion: 10:00:00+05:30 -> 04:30:00 UTC
    assert df["timestamp"].iloc[0].hour == 4
    assert df["timestamp"].iloc[0].minute == 30
    assert str(df["timestamp"].dt.tz) == "UTC"
