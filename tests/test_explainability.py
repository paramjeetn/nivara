"""Unit tests for the Anomaly Explainability Layer (TASK-11)."""
from datetime import datetime
import pytest
import pandas as pd

from campus_energy.schemas import Alert, Severity, AlertStatus
from campus_energy.analytics.explainability import explain_alert, get_detector_type


def test_high_consumption_explainability():
    """Test explainability output for high consumption statistical anomalies."""
    alert = Alert(
        alert_id="ALT-HIGH01",
        detected_at=datetime(2026, 3, 10, 14, 0),  # 2:00 PM Tuesday (weekday)
        asset_id="EQ-HVAC-01",
        building_id="BLD-ENG",
        zone_id="ZN-101",
        anomaly_type="high_consumption",
        severity=Severity.CRITICAL,
        actual_value=7.10,
        expected_value=4.20,
        deviation=2.90,
        evidence="Power exceeded baseline by 2.90 kW for 45 min.",
        recommendation="Inspect equipment for efficiency degradation.",
        confidence=0.92,
    )

    context = {
        "time": "2 PM",
        "day_type": "weekday",
        "outdoor_temperature_c": 34.0,
        "occupancy": 45,
    }

    result = explain_alert(alert, context_row=context)

    assert result["detector_type"] == "Statistical Baseline (Ridge Regression)"
    assert "Expected 4.20 kW at 2 PM (weekday, 34°C, 45 occupants)." in result["why_headline"]
    assert "Actual draw was 7.10 kW (+2.90 kW, +69.0% above baseline)." in result["why_headline"]
    assert isinstance(result["contributing_factors"], list)
    assert len(result["contributing_factors"]) >= 3
    for factor in result["contributing_factors"]:
        assert "factor" in factor
        assert "status" in factor
        assert "detail" in factor
    assert "Thermostat setpoint" in result["probable_root_cause"]
    assert "Check BAS override logs" in result["recommended_verification"]
    assert result["baseline_context"]["expected_kw"] == 4.20
    assert result["baseline_context"]["actual_kw"] == 7.10
    assert result["baseline_context"]["deviation_kw"] == 2.90


def test_unoccupied_operation_explainability():
    """Test explainability output for unoccupied operation rule."""
    alert = Alert(
        alert_id="ALT-UNOCC01",
        detected_at=datetime(2026, 3, 10, 22, 30),
        asset_id="EQ-LT-101",
        building_id="BLD-LIB",
        zone_id="ZN-201",
        anomaly_type="unoccupied_operation",
        severity=Severity.HIGH,
        actual_value=3.50,
        expected_value=0.0,
        deviation=3.50,
        evidence="LIGHTING ran at 3.50 kW with zero occupancy for 45 min.",
        recommendation="Enforce auto-off policy via BMS schedule.",
        confidence=0.90,
    )

    result = explain_alert(alert)

    assert result["detector_type"] == "Operational BMS Rule"
    assert "Asset ran drawing 3.50 kW while zone schedule and occupancy sensor reported 0 occupants" in result["why_headline"]
    assert "LIGHTING ran at 3.50 kW with zero occupancy" in result["why_headline"]
    assert "BMS schedule override" in result["probable_root_cause"]
    assert "Inspect physical wall toggle switches" in result["recommended_verification"]


def test_pump_no_flow_explainability():
    """Test explainability output for hydraulic pump dry-run rule."""
    alert = Alert(
        alert_id="ALT-PUMP01",
        detected_at=datetime(2026, 3, 10, 11, 15),
        asset_id="EQ-PUMP-02",
        building_id="BLD-UTL",
        anomaly_type="pump_no_flow",
        severity=Severity.CRITICAL,
        actual_value=5.50,
        expected_value=100.0,
        evidence="Pump drawing power but reporting 0 flow.",
        recommendation="Dispatch maintenance immediately. Risk of pump burnout.",
        confidence=0.95,
    )

    result = explain_alert(alert)

    assert result["detector_type"] == "Hydraulic Rule"
    assert "Hydraulic motor was energized at 5.50 kW, but flowmeter recorded 0.0 LPM — severe cavitation or dry-run risk." in result["why_headline"]
    assert "Closed discharge/suction isolation valve" in result["probable_root_cause"]
    assert "Verify manual valve positions" in result["recommended_verification"]


def test_sensor_dropout_explainability():
    """Test explainability output for sensor dropout rule."""
    alert = Alert(
        alert_id="ALT-DROP01",
        detected_at=datetime(2026, 3, 10, 8, 45),
        asset_id="MTR-BLD-01",
        building_id="BLD-ENG",
        anomaly_type="sensor_dropout",
        severity=Severity.LOW,
        evidence="Sensor offline / missing data for 25 min.",
        recommendation="Check sensor connectivity and meter communication bus.",
        confidence=0.99,
    )

    result = explain_alert(alert)

    assert result["detector_type"] == "Telemetry Data Quality Rule"
    assert "Sensor telemetry dropped below health heartbeat with quality flag set to offline/missing for extended interval." in result["why_headline"]
    assert "RS-485 Modbus" in result["probable_root_cause"]
    assert "Inspect RS-485 twisted-pair termination" in result["recommended_verification"]


def test_ml_multivariate_anomaly_explainability():
    """Test explainability output for Isolation Forest multivariate anomaly."""
    alert = Alert(
        alert_id="ALT-ML01",
        detected_at=datetime(2026, 3, 10, 16, 20),
        asset_id="EQ-HVAC-02",
        building_id="BLD-ENG",
        anomaly_type="ml_multivariate_anomaly",
        severity=Severity.HIGH,
        actual_value=12.40,
        evidence="Isolation Forest flagged 6 anomalous readings (avg anomaly score: 0.742).",
        recommendation="Investigate asset telemetry pattern.",
        confidence=0.88,
    )

    result = explain_alert(alert)

    assert result["detector_type"] == "Multivariate Isolation Forest (ML)"
    assert "Isolation Forest flagged multivariate outlier across power, 3h rolling mean, standard deviation, and diurnal hour features" in result["why_headline"]
    assert "confidence: 88%" in result["why_headline"]
    assert "Unscheduled multi-equipment co-activation" in result["probable_root_cause"]
    assert "Inspect VFD frequency trend" in result["recommended_verification"]


def test_explain_dict_input():
    """Test that explain_alert handles plain dictionary inputs and Series."""
    dict_alert = {
        "alert_id": "ALT-DICT01",
        "anomaly_type": "high_consumption",
        "actual_value": 8.0,
        "expected_value": 5.0,
        "deviation": 3.0,
        "evidence": "High draw",
        "severity": "high",
        "confidence": 0.85,
        "detected_at": "2026-03-10T14:00:00",
        "asset_id": "EQ-HVAC-01",
        "building_id": "BLD-ENG",
    }

    result = explain_alert(dict_alert)
    assert result["detector_type"] == "Statistical Baseline (Ridge Regression)"
    assert "Expected 5.00 kW" in result["why_headline"]
    assert "Actual draw was 8.00 kW" in result["why_headline"]
    assert result["baseline_context"]["deviation_pct"] == 60.0


def test_edge_cases():
    """Test edge cases such as zero expected value, None values, and fallback anomaly types."""
    # Zero expected value
    alert_zero_exp = {
        "anomaly_type": "high_consumption",
        "actual_value": 2.5,
        "expected_value": 0.0,
        "detected_at": None,
    }
    res_zero = explain_alert(alert_zero_exp)
    assert res_zero["baseline_context"]["deviation_pct"] == 0.0
    assert "Expected 0.00 kW" in res_zero["why_headline"]

    # Unknown anomaly type fallback
    alert_unknown = {
        "anomaly_type": "unexpected_power_surge",
        "actual_value": 15.0,
        "expected_value": 5.0,
        "evidence": "Sudden surge",
    }
    res_unknown = explain_alert(alert_unknown)
    assert "Anomaly detected for asset (unexpected power surge)" in res_unknown["why_headline"]
    assert len(res_unknown["contributing_factors"]) >= 2
    assert res_unknown["probable_root_cause"] is not None
    assert res_unknown["recommended_verification"] is not None

    # Rated power fallback from asset_info for pump_no_flow with 0 actual_value
    pump_alert = {
        "anomaly_type": "pump_no_flow",
        "actual_value": 0.0,
    }
    res_pump = explain_alert(pump_alert, asset_info={"rated_power_kw": 7.5})
    assert "Hydraulic motor was energized at 7.50 kW" in res_pump["why_headline"]
