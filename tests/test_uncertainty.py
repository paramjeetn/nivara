"""Tests for Step 5: Uncertainty, Confidence, and Calibrated Prediction Intervals."""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from campus_energy.analytics.baseline import ExpectedConsumptionModel
from campus_energy.analytics.forecasting import (
    generate_24h_forecast,
    compute_forecast_savings_range,
)
from campus_energy.analytics.anomalies import (
    calculate_confidence_score,
    determine_alert_severity,
)
from campus_energy.analytics.workorders import (
    estimate_energy_waste_range,
    get_work_order_summary,
)
from campus_energy.analytics.esg import calculate_esg_metrics
from campus_energy.eval.forecast_eval import compute_metrics
from campus_energy.schemas import Severity


# ─────────────────────────────────────────────────────────────────────────────
# 1. Forecast Prediction Intervals
# ─────────────────────────────────────────────────────────────────────────────
def test_expected_consumption_model_prediction_intervals():
    """Verify ExpectedConsumptionModel computes empirical bounds satisfying 0 <= lower <= yhat <= upper."""
    model = ExpectedConsumptionModel()
    n_points = 200
    dates = pd.date_range("2025-09-01", periods=n_points, freq="1h")
    # Base pattern + noise
    base_power = 20.0 + 8.0 * np.sin(np.linspace(0, 10 * np.pi, n_points))
    noise = np.random.normal(0, 2.0, n_points)
    train_df = pd.DataFrame({
        "timestamp": dates,
        "power_kw": np.clip(base_power + noise, 1.0, None),
        "hour": dates.hour,
        "day_of_week": dates.dayofweek,
        "outdoor_temperature_c": 30.0 + 5.0 * np.sin(np.linspace(0, 5 * np.pi, n_points)),
        "occupancy": [10] * n_points,
    })

    model.fit(train_df, lower_quantile=10.0, upper_quantile=90.0)
    assert model._is_fitted
    assert len(model.residuals_) == n_points

    test_dates = pd.date_range("2025-09-10", periods=24, freq="1h")
    test_df = pd.DataFrame({
        "timestamp": test_dates,
        "hour": test_dates.hour,
        "day_of_week": test_dates.dayofweek,
        "outdoor_temperature_c": 32.0,
        "occupancy": 10,
    })

    yhat, lower, upper = model.predict(test_df, return_intervals=True)

    # Monotonicity checks
    assert (lower >= 0.0).all(), "Lower prediction interval must be non-negative"
    assert (lower <= yhat).all(), "Expected yhat_lower <= yhat"
    assert (yhat <= upper).all(), "Expected yhat <= yhat_upper"
    assert (upper >= lower).all(), "Expected yhat_upper >= yhat_lower"


def test_generate_24h_forecast_propagates_intervals():
    """Verify generate_24h_forecast populates prediction interval and scenario columns correctly."""
    dates = pd.date_range("2025-09-01", periods=288 * 22, freq="5min")
    power = 40.0 + 15.0 * np.sin(np.linspace(0, 44 * np.pi, len(dates)))
    tel_df = pd.DataFrame({
        "timestamp": dates,
        "asset_id": "MTR-BLD-01",
        "power_kw": np.clip(power, 5.0, None),
    })
    ctx_df = pd.DataFrame({
        "timestamp": dates,
        "outdoor_temperature_c": 32.0,
        "holiday": False,
    })
    assets_df = pd.DataFrame({
        "asset_id": ["MTR-BLD-01"],
        "asset_type": ["meter"],
    })

    fc = generate_24h_forecast(
        telemetry_df=tel_df,
        context_df=ctx_df,
        assets_df=assets_df,
        ac_policy_reduction_pct=20.0,
        occupancy_reduction_pct=15.0,
        solar_pv_kwp=50.0,
        led_upgrade=True,
    )

    assert not fc.empty
    assert len(fc) == 288
    required_cols = [
        "predicted_power_kw",
        "lower_bound",
        "upper_bound",
        "baseline_lower",
        "baseline_upper",
        "scenario_power_kw",
        "scenario_lower",
        "scenario_upper",
    ]
    for c in required_cols:
        assert c in fc.columns, f"Missing required forecast column: {c}"

    # Verify baseline interval ordering
    assert (fc["baseline_lower"] >= 0.0).all()
    assert (fc["baseline_lower"] <= fc["predicted_power_kw"]).all()
    assert (fc["predicted_power_kw"] <= fc["baseline_upper"]).all()

    # Verify scenario interval ordering
    assert (fc["scenario_lower"] >= 0.0).all()
    assert (fc["scenario_lower"] <= fc["scenario_power_kw"]).all()
    assert (fc["scenario_power_kw"] <= fc["scenario_upper"]).all()

    # Scenario savings should reduce power demand
    assert (fc["scenario_power_kw"] <= fc["predicted_power_kw"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Empirical Interval Coverage
# ─────────────────────────────────────────────────────────────────────────────
def test_empirical_interval_coverage_calculation():
    """Verify empirical prediction interval coverage metric calculates actual observation percentage."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    # Points 1, 2, 3, 4 fall within bounds; point 5 falls outside (50 > 48)
    y_lower = np.array([8.0, 18.0, 28.0, 38.0, 40.0])
    y_upper = np.array([12.0, 22.0, 32.0, 42.0, 48.0])

    metrics = compute_metrics(y_true=y_true, y_pred=y_true, y_lower=y_lower, y_upper=y_upper)
    assert metrics["coverage_pct"] == 80.0  # 4 out of 5 covered = 80.0%

    # 100% coverage
    y_upper_wide = np.array([15.0, 25.0, 35.0, 45.0, 55.0])
    metrics_all = compute_metrics(y_true=y_true, y_pred=y_true, y_lower=y_lower, y_upper=y_upper_wide)
    assert metrics_all["coverage_pct"] == 100.0

    # 0% coverage
    y_lower_miss = np.array([100.0] * 5)
    y_upper_miss = np.array([200.0] * 5)
    metrics_none = compute_metrics(y_true=y_true, y_pred=y_true, y_lower=y_lower_miss, y_upper=y_upper_miss)
    assert metrics_none["coverage_pct"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Anomaly Severity & Confidence Formulations
# ─────────────────────────────────────────────────────────────────────────────
def test_anomaly_confidence_scoring_bounds_and_agreement():
    """Verify confidence score is bounded in [0.10, 1.00] and increases with multi-layer agreement."""
    # Base confidence tests by detector type
    conf_rule_single = calculate_confidence_score("rule", has_layer_agreement=False, deviation=0.0)
    conf_stat_single = calculate_confidence_score("statistical", has_layer_agreement=False, deviation=0.0)
    conf_ml_single = calculate_confidence_score("isolation_forest", has_layer_agreement=False, deviation=0.0)

    assert conf_rule_single == 0.85
    assert conf_stat_single == 0.65
    assert conf_ml_single == 0.50

    # Multi-layer agreement bonus adds +0.15
    conf_stat_agreed = calculate_confidence_score("statistical", has_layer_agreement=True, deviation=0.0)
    assert pytest.approx(conf_stat_agreed, 0.01) == 0.80  # 0.65 + 0.15
    assert conf_stat_agreed > conf_stat_single

    # Residual magnitude scaling
    conf_stat_high_dev = calculate_confidence_score(
        "statistical", has_layer_agreement=False, deviation=30.0, baseline_power=20.0
    )
    assert conf_stat_high_dev > conf_stat_single

    # Bounded in [0.10, 1.00]
    conf_extreme_high = calculate_confidence_score("rule", has_layer_agreement=True, deviation=1000.0, baseline_power=10.0)
    assert conf_extreme_high <= 1.00

    conf_extreme_low = calculate_confidence_score("unknown", has_layer_agreement=False, deviation=0.0)
    assert conf_extreme_low >= 0.10


def test_determine_alert_severity():
    """Verify impact- and cost-based severity grading according to specification."""
    # Critical: deviation >= critical threshold (50.0 kW)
    assert determine_alert_severity("high_consumption", deviation=60.0, crit_cutoff_kw=50.0) == Severity.CRITICAL

    # High: pump dry-run or after-hours running of high-draw HVAC
    assert determine_alert_severity("pump_no_flow") == Severity.HIGH
    assert determine_alert_severity("unoccupied_operation", asset_type="hvac", deviation=22.0) == Severity.HIGH

    # Medium: lighting left on
    assert determine_alert_severity("unoccupied_operation", asset_type="lighting", deviation=5.0) == Severity.MEDIUM
    assert determine_alert_severity("lights_left_on") == Severity.MEDIUM

    # Low: sensor dropout
    assert determine_alert_severity("sensor_dropout") == Severity.LOW


# ─────────────────────────────────────────────────────────────────────────────
# 4. Standardized Savings Ranges {low, expected, high}
# ─────────────────────────────────────────────────────────────────────────────
def test_forecast_savings_range():
    """Verify compute_forecast_savings_range produces low <= expected <= high for kWh, INR, and CO2."""
    dates = pd.date_range("2025-09-01", periods=288, freq="5min")
    pred = np.array([50.0] * 288)
    scen = np.array([40.0] * 288)  # 10 kW continuous reduction = 240 kWh expected
    df = pd.DataFrame({
        "timestamp": dates,
        "predicted_power_kw": pred,
        "lower_bound": pred * 0.9,
        "upper_bound": pred * 1.1,
        "baseline_lower": pred * 0.9,
        "baseline_upper": pred * 1.1,
        "scenario_power_kw": scen,
        "scenario_lower": scen * 0.9,
        "scenario_upper": scen * 1.1,
    })

    res = compute_forecast_savings_range(df, cost_per_kwh=8.0, grid_emission_factor=0.8)

    for metric in ("kwh", "cost_inr", "co2_kg"):
        assert metric in res
        low = res[metric]["low"]
        exp = res[metric]["expected"]
        high = res[metric]["high"]
        assert low >= 0.0
        assert low <= exp, f"{metric}: expected low ({low}) <= exp ({exp})"
        assert exp <= high, f"{metric}: expected exp ({exp}) <= high ({high})"


def test_workorder_savings_range():
    """Verify estimate_energy_waste_range satisfies low <= expected <= high."""
    ranges = estimate_energy_waste_range(rated_power_kw=15.0, severity="high")

    assert ranges["kwh"]["low"] <= ranges["kwh"]["expected"] <= ranges["kwh"]["high"]
    assert ranges["inr"]["low"] <= ranges["inr"]["expected"] <= ranges["inr"]["high"]

    wo_df = pd.DataFrame([{
        "wo_id": "WO-001",
        "severity": "high",
        "status": "Open",
        "est_waste_kwh_day": 100.0,
        "est_waste_inr_day": 800.0,
        "priority_score": 85.0,
    }])
    summary = get_work_order_summary(wo_df)
    assert "daily_waste_kwh_range" in summary
    assert summary["daily_waste_kwh_range"]["low"] <= summary["daily_waste_kwh_range"]["expected"] <= summary["daily_waste_kwh_range"]["high"]
    assert "⚠️ Decision-support estimate, not an official measurement." in summary["disclaimer"]


def test_esg_savings_range_and_disclaimer():
    """Verify calculate_esg_metrics produces standardized savings ranges and decision-support disclaimer."""
    # Synthetic telemetry and events
    dates = pd.date_range("2025-09-01", periods=288 * 5, freq="5min", tz="UTC")
    tel_df = pd.DataFrame({
        "timestamp": dates,
        "reading_type": "meter",
        "asset_id": "MTR-01",
        "interval_energy_kwh": [5.0] * len(dates),
    })
    events_df = pd.DataFrame([{
        "asset_id": "MTR-01",
        "start_time": dates[10],
        "end_time": dates[50],
    }])

    res = calculate_esg_metrics(tel_df, events_df, floor_area_m2=5000.0, facility_type="university")
    kpis = res["kpis"]

    assert "avoidable_savings_range" in kpis
    assert "napcc_savings_range" in kpis
    assert "disclaimer" in kpis
    assert "Decision-support estimate" in kpis["disclaimer"]

    # Verify range bounds
    av_kwh = kpis["avoidable_savings_range"]["kwh"]
    assert av_kwh["low"] <= av_kwh["expected"] <= av_kwh["high"]
    av_cost = kpis["avoidable_savings_range"]["cost_inr"]
    assert av_cost["low"] <= av_cost["expected"] <= av_cost["high"]
