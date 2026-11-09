"""Data Quality Engine for Nivara Campus Energy platform.

Evaluates raw or ingested telemetry datasets against platform data quality rules:
- Missing intervals and gaps in sampling timeline
- Duplicate readings (same asset + timestamp)
- Stuck / frozen sensors (identical values for > N consecutive intervals)
- Extreme physical outliers (negative power, power factor > 1.0, voltage > 300V)
- Temporal completeness and asset coverage %

Generates a structured DataQualityReport and tags individual records with QualityFlag
and SensorStatus contracts.
"""
from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd

from campus_energy.schemas import DataQualityReport, QualityFlag, SensorStatus

logger = logging.getLogger(__name__)


class DataQualityEngine:
    """Configurable data quality evaluation engine for time-series telemetry."""

    def __init__(
        self,
        expected_interval_minutes: int | None = None,
        stuck_threshold_steps: int = 12,
        outlier_voltage_max_v: float = 300.0,
        outlier_voltage_min_v: float = 0.0,
        outlier_pf_max: float = 1.0,
        outlier_pf_min: float = 0.0,
        outlier_power_max_kw: float = 10000.0,
        outlier_power_min_kw: float = 0.0,
    ) -> None:
        self.expected_interval_minutes = expected_interval_minutes
        self.stuck_threshold_steps = stuck_threshold_steps
        self.outlier_voltage_max_v = outlier_voltage_max_v
        self.outlier_voltage_min_v = outlier_voltage_min_v
        self.outlier_pf_max = outlier_pf_max
        self.outlier_pf_min = outlier_pf_min
        self.outlier_power_max_kw = outlier_power_max_kw
        self.outlier_power_min_kw = outlier_power_min_kw

    def evaluate(self, df: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
        """Evaluate a telemetry dataframe against all quality checks.

        Returns:
            Tuple of (tagged_dataframe, data_quality_report).
        """
        if df.empty:
            report = DataQualityReport(
                total_records=0,
                quality_score=0.0,
                missing_intervals_count=0,
                duplicate_count=0,
                stuck_sensor_count=0,
                outlier_count=0,
                asset_coverage_pct=0.0,
                confidence_penalty=-0.40,
                recommendations=["Empty telemetry dataset: No records available for quality analysis."],
            )
            tagged = df.copy()
            if "quality_flag" not in tagged.columns:
                tagged["quality_flag"] = []
            if "sensor_status" not in tagged.columns:
                tagged["sensor_status"] = []
            return tagged, report

        tagged = df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(tagged["timestamp"]):
            tagged["timestamp"] = pd.to_datetime(tagged["timestamp"])

        # Default asset_id if missing
        if "asset_id" not in tagged.columns:
            tagged["asset_id"] = "EXTERNAL_ASSET"

        # Initialize quality tags
        tagged["quality_flag"] = QualityFlag.VALID.value
        tagged["sensor_status"] = SensorStatus.HEALTHY.value

        total_records = len(tagged)

        # -------------------------------------------------------------
        # 1. Duplicate check (same asset + timestamp)
        # -------------------------------------------------------------
        dup_mask = tagged.duplicated(subset=["asset_id", "timestamp"], keep="first")
        duplicate_count = int(dup_mask.sum())

        # -------------------------------------------------------------
        # 2. Extreme Physical Outliers
        # -------------------------------------------------------------
        outlier_mask = pd.Series(False, index=tagged.index)

        # Power outliers (e.g. negative power or impossible spike)
        if "power_kw" in tagged.columns:
            pwr_series = pd.to_numeric(tagged["power_kw"], errors="coerce")
            neg_power = pwr_series < self.outlier_power_min_kw
            spike_power = pwr_series > self.outlier_power_max_kw
            outlier_mask = outlier_mask | neg_power | spike_power

        # Power factor outliers (0.0 to 1.0)
        if "power_factor" in tagged.columns:
            pf_series = pd.to_numeric(tagged["power_factor"], errors="coerce")
            pf_invalid = pf_series.notna() & ((pf_series < self.outlier_pf_min) | (pf_series > self.outlier_pf_max))
            outlier_mask = outlier_mask | pf_invalid

        # Voltage outliers (> 300V single-phase / line-to-neutral or < 0)
        if "voltage_v" in tagged.columns:
            volt_series = pd.to_numeric(tagged["voltage_v"], errors="coerce")
            volt_invalid = volt_series.notna() & ((volt_series < self.outlier_voltage_min_v) | (volt_series > self.outlier_voltage_max_v))
            outlier_mask = outlier_mask | volt_invalid

        outlier_count = int(outlier_mask.sum())

        # -------------------------------------------------------------
        # 3. Stuck Sensors (identical consecutive readings >= N steps)
        # -------------------------------------------------------------
        stuck_mask = pd.Series(False, index=tagged.index)
        unique_assets = tagged["asset_id"].unique()

        # Measurement column to check for stuck sensor
        val_col = "power_kw" if "power_kw" in tagged.columns else None

        if val_col is not None:
            for aid in unique_assets:
                asset_indices = tagged[tagged["asset_id"] == aid].sort_values("timestamp").index
                if len(asset_indices) >= self.stuck_threshold_steps:
                    val_s = pd.to_numeric(tagged.loc[asset_indices, val_col], errors="coerce")
                    # Compare each element with previous
                    is_equal = np.isclose(val_s.values, np.roll(val_s.values, 1), atol=1e-5, equal_nan=False)
                    is_equal[0] = False
                    run_ids = (~is_equal).cumsum()
                    run_counts = pd.Series(run_ids).value_counts()
                    stuck_runs = set(run_counts[run_counts >= self.stuck_threshold_steps].index)
                    if stuck_runs:
                        asset_stuck = pd.Series(run_ids).isin(stuck_runs).values & val_s.notna().values
                        stuck_mask.loc[asset_indices[asset_stuck]] = True

        stuck_sensor_count = int(stuck_mask.sum())

        # -------------------------------------------------------------
        # 4. Missing Intervals and Timeline Gaps
        # -------------------------------------------------------------
        missing_intervals_count = 0
        total_expected_records = 0

        # Estimate global or default interval
        all_diffs = []
        for aid in unique_assets:
            sub = tagged[tagged["asset_id"] == aid].sort_values("timestamp")
            if len(sub) > 1:
                diff_mins = sub["timestamp"].diff().dt.total_seconds() / 60.0
                pos_diffs = diff_mins[diff_mins > 0]
                all_diffs.extend(pos_diffs.tolist())

        if self.expected_interval_minutes is not None:
            nominal_interval = float(self.expected_interval_minutes)
        elif all_diffs:
            nominal_interval = float(pd.Series(all_diffs).median())
            if nominal_interval <= 0 or np.isnan(nominal_interval):
                nominal_interval = 15.0
        else:
            nominal_interval = 15.0

        for aid in unique_assets:
            sub = tagged[tagged["asset_id"] == aid].sort_values("timestamp")
            if len(sub) > 1:
                t_min = sub["timestamp"].min()
                t_max = sub["timestamp"].max()
                span_mins = (t_max - t_min).total_seconds() / 60.0
                exp_steps = int(round(span_mins / max(nominal_interval, 1.0))) + 1
                total_expected_records += exp_steps

                diff_mins = sub["timestamp"].diff().dt.total_seconds() / 60.0
                gap_diffs = diff_mins[diff_mins > 1.5 * nominal_interval]
                for d in gap_diffs:
                    missed = int(round(d / nominal_interval)) - 1
                    if missed > 0:
                        missing_intervals_count += missed
            else:
                total_expected_records += 1

        # -------------------------------------------------------------
        # 5. Asset Coverage %
        # -------------------------------------------------------------
        if total_expected_records > 0:
            coverage_pct = round(min(100.0, max(0.0, (total_records / total_expected_records) * 100.0)), 2)
        else:
            coverage_pct = 100.0

        # -------------------------------------------------------------
        # 6. Apply Quality Tags to Dataframe
        # -------------------------------------------------------------
        # Apply in priority order: STALE -> OUT_OF_RANGE -> DUPLICATED
        if stuck_sensor_count > 0:
            tagged.loc[stuck_mask, "quality_flag"] = QualityFlag.STALE.value
            tagged.loc[stuck_mask, "sensor_status"] = SensorStatus.DEGRADED.value

        if outlier_count > 0:
            tagged.loc[outlier_mask, "quality_flag"] = QualityFlag.OUT_OF_RANGE.value
            tagged.loc[outlier_mask, "sensor_status"] = SensorStatus.DEGRADED.value

        if duplicate_count > 0:
            tagged.loc[dup_mask, "quality_flag"] = QualityFlag.DUPLICATED.value
            tagged.loc[dup_mask, "sensor_status"] = SensorStatus.DEGRADED.value

        # -------------------------------------------------------------
        # 7. Quality Score and Confidence Penalty Calculation
        # -------------------------------------------------------------
        total_defects = duplicate_count + outlier_count + stuck_sensor_count + missing_intervals_count
        denom = max(1, total_records + missing_intervals_count)
        defect_ratio = min(1.0, total_defects / denom)
        quality_score = round(max(0.0, min(100.0, 100.0 * (1.0 - defect_ratio))), 2)

        # Calibrate penalty
        if quality_score >= 95.0:
            confidence_penalty = 0.0
        elif quality_score >= 80.0:
            confidence_penalty = -0.10
        elif quality_score >= 65.0:
            confidence_penalty = -0.20
        elif quality_score >= 50.0:
            confidence_penalty = -0.30
        else:
            confidence_penalty = -0.40

        # -------------------------------------------------------------
        # 8. Recommendations Generation
        # -------------------------------------------------------------
        recommendations: list[str] = []
        if missing_intervals_count > 0:
            recommendations.append(
                f"Detected {missing_intervals_count} missing interval gaps (cadence: {nominal_interval:.0f} min). "
                "Re-index timestamp cadence and check telemetry network connectivity."
            )
        if duplicate_count > 0:
            recommendations.append(
                f"Found {duplicate_count} duplicate timestamp readings. "
                "Deduplicate ingestion pipeline to prevent double-counting."
            )
        if stuck_sensor_count > 0:
            recommendations.append(
                f"Identified {stuck_sensor_count} consecutive stuck sensor readings (threshold: {self.stuck_threshold_steps} steps). "
                "Inspect sensors on affected assets for frozen telemetry."
            )
        if outlier_count > 0:
            recommendations.append(
                f"Detected {outlier_count} physical outliers (e.g., negative power, PF > 1.0, or voltage > 300V). "
                "Recalibrate meter sensors."
            )
        if coverage_pct < 85.0:
            recommendations.append(
                f"Asset temporal coverage is degraded at {coverage_pct:.1f}%. "
                "Verify that all deployed sub-meters are streaming actively across the time window."
            )
        if not recommendations:
            recommendations.append("Data quality is excellent (>= 95%). Telemetry meets all operational fidelity standards.")

        report = DataQualityReport(
            total_records=total_records,
            quality_score=quality_score,
            missing_intervals_count=missing_intervals_count,
            duplicate_count=duplicate_count,
            stuck_sensor_count=stuck_sensor_count,
            outlier_count=outlier_count,
            asset_coverage_pct=coverage_pct,
            confidence_penalty=confidence_penalty,
            recommendations=recommendations,
        )

        return tagged, report


def evaluate_telemetry_quality(
    df: pd.DataFrame,
    expected_interval_minutes: int | None = None,
    stuck_threshold_steps: int = 12,
    outlier_voltage_max_v: float = 300.0,
    outlier_voltage_min_v: float = 0.0,
    outlier_pf_max: float = 1.0,
    outlier_pf_min: float = 0.0,
    outlier_power_max_kw: float = 10000.0,
    outlier_power_min_kw: float = 0.0,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Functional convenience entrypoint for evaluating telemetry quality."""
    engine = DataQualityEngine(
        expected_interval_minutes=expected_interval_minutes,
        stuck_threshold_steps=stuck_threshold_steps,
        outlier_voltage_max_v=outlier_voltage_max_v,
        outlier_voltage_min_v=outlier_voltage_min_v,
        outlier_pf_max=outlier_pf_max,
        outlier_pf_min=outlier_pf_min,
        outlier_power_max_kw=outlier_power_max_kw,
        outlier_power_min_kw=outlier_power_min_kw,
    )
    return engine.evaluate(df)
