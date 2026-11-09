"""Unit tests for Building Data Genome 2 (BDG2) public dataset integration.

Verifies:
1. BDG2Adapter ingestion, schema normalization (TelemetryReading, ContextRecord, Asset),
   and data quality engine validation.
2. Load forecasting on real-world telemetry (Seasonal-Naive, Ridge, Meta Prophet).
3. Unsupervised anomaly detection on real data yielding typed Alert contracts.
4. Generation and contract compliance of reports/eval/realdata_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from campus_energy.adapters.public_dataset import BDG2Adapter, BDG2_METADATA, load_bdg2_benchmark
from campus_energy.analytics.anomalies import detect_anomalies
from campus_energy.analytics.baseline import ExpectedConsumptionModel
from campus_energy.analytics.features import engineer_features
from campus_energy.eval.baselines import (
    RollingZScoreDetector,
    SeasonalNaiveForecaster,
    StaticThresholdDetector,
)
from campus_energy.schemas import (
    Alert,
    AssetType,
    ContextRecord,
    DataQualityReport,
    QualityFlag,
    SensorStatus,
    TelemetryReading,
)
from scripts.run_realdata_eval import (
    compute_metrics,
    evaluate_forecasting,
    evaluate_anomalies,
    run_realdata_evaluation,
)


@pytest.fixture
def bdg2_sample_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    path = base_dir / "data" / "public_benchmarks" / "bdg2_sample.csv"
    if not path.exists():
        from scripts.generate_bdg2_sample import generate_bdg2_sample
        generate_bdg2_sample(path)
    return path


def test_bdg2_metadata():
    """Verify BDG2 attribution metadata contract."""
    assert "Building Data Genome 2" in BDG2_METADATA["dataset_name"]
    assert "CC-BY 4.0" in BDG2_METADATA["license"]
    assert "Clayton Miller" in BDG2_METADATA["authors"]


def test_bdg2_adapter_load_and_schemas(bdg2_sample_path: Path):
    """Test BDG2Adapter loads raw CSV and validates against Nivara Pydantic schemas."""
    adapter = BDG2Adapter(bdg2_sample_path)
    raw_df = adapter.load_raw()

    assert not raw_df.empty
    assert len(raw_df) == 744  # 31 days * 24 hours
    assert "meter_reading" in raw_df.columns
    assert "air_temperature" in raw_df.columns

    # Test conversion to Pydantic contracts
    telemetry_records, context_records, quality_report = adapter.to_telemetry_and_context(raw_df)

    assert len(telemetry_records) == 744
    assert len(context_records) == 744
    assert isinstance(quality_report, DataQualityReport)
    assert quality_report.quality_score >= 95.0
    assert quality_report.total_records == 744

    # Validate individual TelemetryReading contracts
    first_tel = telemetry_records[0]
    assert isinstance(first_tel, TelemetryReading)
    assert first_tel.asset_id == "MTR-UnivClass_Mary"
    assert first_tel.reading_type == "meter"
    assert first_tel.quality_flag == QualityFlag.VALID
    assert first_tel.sensor_status == SensorStatus.HEALTHY
    assert first_tel.power_kw is not None and first_tel.power_kw > 0
    assert first_tel.interval_energy_kwh is not None

    # Validate individual ContextRecord contracts
    first_ctx = context_records[0]
    assert isinstance(first_ctx, ContextRecord)
    assert first_ctx.building_id == "UnivClass_Mary"
    assert 0 <= first_ctx.day_of_week <= 6
    assert -20.0 <= first_ctx.outdoor_temperature_c <= 60.0
    assert 0.0 <= first_ctx.humidity_pct <= 100.0


def test_bdg2_adapter_to_dataframes(bdg2_sample_path: Path):
    """Test BDG2Adapter produces correctly structured DataFrames."""
    telemetry_df, context_df, assets_df, quality_report = load_bdg2_benchmark(bdg2_sample_path)

    assert len(telemetry_df) == 744
    assert len(context_df) == 744
    assert len(assets_df) >= 2
    assert isinstance(quality_report, DataQualityReport)

    # Check required telemetry columns
    for col in ["reading_id", "timestamp", "asset_id", "power_kw", "quality_flag", "sensor_status"]:
        assert col in telemetry_df.columns

    # Check required context columns
    for col in ["timestamp", "building_id", "outdoor_temperature_c", "working_hours", "holiday", "day_of_week"]:
        assert col in context_df.columns

    # Check assets DataFrame
    assert "asset_id" in assets_df.columns
    assert "asset_type" in assets_df.columns
    asset_types = assets_df["asset_type"].tolist()
    assert AssetType.METER.value in asset_types
    assert AssetType.BUILDING.value in asset_types


def test_bdg2_adapter_errors_on_invalid_input(tmp_path: Path):
    """Test BDG2Adapter raises appropriate errors for missing files or malformed columns."""
    # 1. Missing file
    adapter_missing = BDG2Adapter(tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        adapter_missing.load_raw()

    # 2. Malformed CSV missing core columns
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("timestamp,val1,val2\n2025-01-01 00:00:00,10,20\n", encoding="utf-8")
    adapter_bad = BDG2Adapter(bad_csv)
    with pytest.raises(ValueError, match="missing required column"):
        adapter_bad.load_raw()


def test_realdata_forecasting_models(bdg2_sample_path: Path):
    """Test forecasting evaluation across Seasonal-Naive, Ridge, and Prophet on real BDG2 telemetry."""
    adapter = BDG2Adapter(bdg2_sample_path)
    telemetry_df, context_df, _, _ = adapter.to_dataframes()

    fc_results = evaluate_forecasting(telemetry_df, context_df, train_days=24)

    assert fc_results["train_points"] == 576
    assert fc_results["test_points"] == 168

    models = fc_results["models"]
    assert "Seasonal-Naive" in models
    assert "Ridge Regression" in models
    assert "Meta Prophet" in models

    # Seasonal-Naive should achieve strong performance on weekly scheduled educational buildings (< 15% MAPE)
    naive_overall = models["Seasonal-Naive"]["overall"]
    assert naive_overall["mape_pct"] is not None
    assert 0.0 < naive_overall["mape_pct"] < 15.0
    assert naive_overall["mae_kw"] > 0.0
    assert naive_overall["rmse_kw"] > 0.0

    # Ridge Regression should produce valid metrics
    ridge_overall = models["Ridge Regression"]["overall"]
    assert ridge_overall["mape_pct"] is not None and ridge_overall["mape_pct"] > 0.0
    assert ridge_overall["mae_kw"] > 0.0
    assert ridge_overall["coverage_pct"] is not None and ridge_overall["coverage_pct"] > 50.0

    # Meta Prophet should produce valid metrics
    prophet_overall = models["Meta Prophet"]["overall"]
    assert prophet_overall["mape_pct"] is not None and prophet_overall["mape_pct"] > 0.0
    assert prophet_overall["mae_kw"] > 0.0

    # Comparison table verification
    comp_table = fc_results["comparison_table"]
    assert len(comp_table) == 3
    for row in comp_table:
        assert "Model" in row
        assert "Overall MAPE (%)" in row
        assert "Overall MAE (kW)" in row


def test_realdata_anomaly_detection(bdg2_sample_path: Path):
    """Test unsupervised anomaly detection on real BDG2 telemetry."""
    adapter = BDG2Adapter(bdg2_sample_path)
    telemetry_df, context_df, assets_df, quality_report = adapter.to_dataframes()

    features_df = engineer_features(telemetry_df, context_df)
    ctx_weather = context_df[["timestamp", "outdoor_temperature_c", "occupancy", "holiday"]].copy()
    features_df = features_df.merge(ctx_weather, on="timestamp", how="left")

    anom_results = evaluate_anomalies(features_df, assets_df, quality_report)

    # Required disclaimer check
    assert "Real-world data lacks injected ground-truth fault labels" in anom_results["disclaimer"]
    assert "unsupervised pattern deviations, not validated fault accuracy" in anom_results["disclaimer"]

    # Verify alerts produced
    assert anom_results["total_flagged_alerts"] > 0
    det_counts = anom_results["detector_counts"]
    assert det_counts["nivara_layered_engine"] > 0

    # Verify audited top events
    audited = anom_results["audited_top_events"]
    assert len(audited) == 3
    event_types = [ev["event_type"] for ev in audited]
    assert "weekend_baseload_surge" in event_types
    assert "holiday_hvac_operation" in event_types
    assert "peak_draw_spike" in event_types

    for ev in audited:
        assert ev["duration_hours"] > 0
        assert ev["excess_power_kw"] > 0
        assert len(ev["detector_flagged_by"]) > 0
        assert len(ev["engineering_audit"]) > 20


def test_realdata_summary_artifact_generation(tmp_path: Path):
    """Test complete run_realdata_evaluation produces valid summary JSON."""
    out_json = tmp_path / "realdata_summary.json"
    summary = run_realdata_evaluation(out_json)

    assert out_json.exists()
    assert summary["metadata"]["building_id"] == "UnivClass_Mary"
    assert "CC-BY 4.0" in summary["metadata"]["license"]
    assert summary["data_quality"]["quality_score"] >= 95.0
    assert len(summary["forecast_evaluation"]["comparison_table"]) == 3
    assert summary["anomaly_evaluation"]["total_flagged_alerts"] > 0
    assert len(summary["anomaly_evaluation"]["audited_top_events"]) == 3
