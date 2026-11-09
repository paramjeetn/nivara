"""Unit tests for the evaluation harness and baselines."""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from campus_energy.eval.baselines import (
    StaticThresholdDetector,
    RollingZScoreDetector,
    SeasonalNaiveForecaster,
    detect_static_threshold,
    detect_rolling_zscore,
    generate_seasonal_naive_forecast,
)
from campus_energy.eval.anomaly_eval import match_alerts_to_ground_truth
from campus_energy.schemas import Alert, Severity


def test_event_matching_hand_built_case():
    """Validate strict TP, FP, FN, precision, recall, and detection delay calculation."""
    # 2 Ground Truth Events
    gt_df = pd.DataFrame([
        {
            "event_id": "EVT-01",
            "asset_id": "EQ-PUMP-01",
            "event_type": "pump_dry_run",
            "start_time": pd.to_datetime("2025-09-01 10:00:00"),
            "end_time": pd.to_datetime("2025-09-01 11:00:00"),
        },
        {
            "event_id": "EVT-02",
            "asset_id": "EQ-HVAC-02",
            "event_type": "hvac_left_on",
            "start_time": pd.to_datetime("2025-09-01 14:00:00"),
            "end_time": pd.to_datetime("2025-09-01 16:00:00"),
        },
    ])

    # 3 Alerts:
    # Alert 1: matches EVT-01 (detected at 10:45 -> 45 min delay)
    # Alert 2: also matches EVT-01 (detected at 11:15 within tolerance -> duplicate TP alert, must not inflate TP)
    # Alert 3: matches nothing (asset EQ-SOLAR-99 has no event -> FP)
    # EVT-02 has no alerts -> FN
    alerts = [
        Alert(
            alert_id="ALT-1",
            detected_at=pd.to_datetime("2025-09-01 10:45:00"),
            event_start_estimate=pd.to_datetime("2025-09-01 10:15:00"),
            asset_id="EQ-PUMP-01",
            building_id="BLD-A",
            zone_id="ZONE-1",
            anomaly_type="pump_no_flow",
            severity=Severity.HIGH,
            evidence="Pump test",
            recommendation="Inspect",
            confidence=0.9,
        ),
        Alert(
            alert_id="ALT-2",
            detected_at=pd.to_datetime("2025-09-01 11:15:00"),
            event_start_estimate=pd.to_datetime("2025-09-01 10:30:00"),
            asset_id="EQ-PUMP-01",
            building_id="BLD-A",
            zone_id="ZONE-1",
            anomaly_type="pump_no_flow",
            severity=Severity.HIGH,
            evidence="Pump test duplicate",
            recommendation="Inspect",
            confidence=0.9,
        ),
        Alert(
            alert_id="ALT-3",
            detected_at=pd.to_datetime("2025-09-01 12:30:00"),
            event_start_estimate=pd.to_datetime("2025-09-01 12:00:00"),
            asset_id="EQ-SOLAR-99",
            building_id="BLD-B",
            zone_id="ZONE-2",
            anomaly_type="static_threshold",
            severity=Severity.MEDIUM,
            evidence="Solar false spike",
            recommendation="None",
            confidence=0.7,
        ),
    ]

    res = match_alerts_to_ground_truth(alerts, gt_df, tolerance_minutes=60, model_name="TestEngine")

    assert res.true_positives == 1, "Expected exactly 1 TP event"
    assert res.false_negatives == 1, "Expected exactly 1 FN event (EVT-02)"
    assert res.false_positives == 1, "Expected exactly 1 FP alert (ALT-3)"
    assert pytest.approx(res.precision, 0.001) == 0.5  # 1 / (1 + 1)
    assert pytest.approx(res.recall, 0.001) == 0.5     # 1 / (1 + 1)
    assert pytest.approx(res.f1_score, 0.001) == 0.5   # 2*(0.5*0.5)/(1.0)
    assert pytest.approx(res.mean_delay_minutes, 0.1) == 45.0  # 10:45 - 10:00 = 45 min
    assert "pump_dry_run" in res.per_fault_type
    assert res.per_fault_type["pump_dry_run"]["detected"] == 1
    assert res.per_fault_type["hvac_left_on"]["detected"] == 0


def test_static_threshold_detector():
    """Verify StaticThresholdDetector detects sustained threshold breaches."""
    base_time = pd.to_datetime("2025-09-01 08:00:00")
    timestamps = [base_time + timedelta(minutes=5 * i) for i in range(10)]
    # Values: 1.0 kW normal, then 3 sustained 5.0 kW spikes, then 1.0 kW normal
    powers = [1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    features_df = pd.DataFrame({
        "timestamp": timestamps,
        "asset_id": ["EQ-TEST"] * 10,
        "power_kw": powers,
    })
    assets_df = pd.DataFrame([{"asset_id": "EQ-TEST", "building_id": "BLD-1", "rated_power_kw": 2.0}])

    # With fixed threshold 3.0 kW and min_consecutive = 3
    alerts = detect_static_threshold(features_df, assets_df, fixed_threshold_kw=3.0, min_consecutive=3)
    assert len(alerts) == 1, "Expected exactly 1 static threshold alert"
    assert alerts[0].asset_id == "EQ-TEST"
    assert alerts[0].anomaly_type == "static_threshold"
    assert alerts[0].event_start_estimate == timestamps[2]
    assert alerts[0].detected_at == timestamps[4]

    # With fixed threshold 6.0 kW (none should breach)
    no_alerts = detect_static_threshold(features_df, assets_df, fixed_threshold_kw=6.0, min_consecutive=3)
    assert len(no_alerts) == 0


def test_rolling_zscore_detector():
    """Verify RollingZScoreDetector detects sudden relative standard deviation spikes."""
    base_time = pd.to_datetime("2025-09-01 00:00:00")
    # 50 steady points at 1.0 kW, then 4 spike points at 15.0 kW, then 10 steady points
    timestamps = [base_time + timedelta(minutes=5 * i) for i in range(64)]
    powers = [1.0] * 50 + [15.0] * 4 + [1.0] * 10

    features_df = pd.DataFrame({
        "timestamp": timestamps,
        "asset_id": ["EQ-TEST"] * 64,
        "power_kw": powers,
    })
    assets_df = pd.DataFrame([{"asset_id": "EQ-TEST", "building_id": "BLD-1", "rated_power_kw": 10.0}])

    detector = RollingZScoreDetector(window=50, z_threshold=3.0, min_consecutive=3)
    alerts = detector.detect(features_df, assets_df)

    assert len(alerts) >= 1
    assert alerts[0].anomaly_type == "rolling_zscore"
    assert alerts[0].asset_id == "EQ-TEST"


def test_seasonal_naive_forecaster():
    """Verify SeasonalNaiveForecaster retrieves exact matching slot from 7 days prior."""
    start_ts = pd.to_datetime("2025-09-01 00:00:00")
    # Generate 14 days of hourly data (336 hours)
    train_hours = 14 * 24
    timestamps = [start_ts + timedelta(hours=i) for i in range(train_hours)]
    # Daily cycle: power = 10 + 5 * sin(2*pi*hour/24)
    powers = [10.0 + 5.0 * np.sin(2 * np.pi * (t.hour / 24.0)) for t in timestamps]

    train_df = pd.DataFrame({
        "timestamp": timestamps,
        "power_kw": powers,
    })

    forecaster = SeasonalNaiveForecaster(seasonal_period_days=7)
    forecaster.fit(train_df)

    # Predict Day 15 (which should match Day 8, exactly 7 days prior)
    target_ts = [start_ts + timedelta(days=14, hours=h) for h in range(24)]
    fc_df = forecaster.predict(target_ts)

    expected_values = [train_df[train_df["timestamp"] == (t - timedelta(days=7))]["power_kw"].iloc[0] for t in target_ts]
    np.testing.assert_allclose(fc_df["predicted_power_kw"].values, expected_values, rtol=1e-5)


def test_seasonal_naive_daily_fallback():
    """Verify SeasonalNaiveForecaster falls back to t - 1 day when t - 7 days is missing."""
    start_ts = pd.to_datetime("2025-09-01 00:00:00")
    # Only 2 days of history (not enough for 7 days)
    timestamps = [start_ts + timedelta(hours=i) for i in range(48)]
    powers = [5.0 + (t.hour % 6) for t in timestamps]

    train_df = pd.DataFrame({"timestamp": timestamps, "power_kw": powers})
    forecaster = SeasonalNaiveForecaster(seasonal_period_days=7, fallback_period_days=1)
    forecaster.fit(train_df)

    # Predict Day 3 hour 10 -> t - 7 days missing, but t - 1 day (Day 2 hour 10) is present
    target_ts = [start_ts + timedelta(days=2, hours=10)]
    pred = forecaster.predict(target_ts)

    day2_hr10_val = train_df[train_df["timestamp"] == (start_ts + timedelta(days=1, hours=10))]["power_kw"].iloc[0]
    assert pytest.approx(pred["predicted_power_kw"].iloc[0], 0.01) == day2_hr10_val
