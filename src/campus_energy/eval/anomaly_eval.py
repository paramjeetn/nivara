"""Anomaly detection evaluation module.

Benchmarks the full layered engine against individual ablated layers and naive baselines
using injected ground-truth fault events (anomaly_ground_truth.csv and events.csv).

Event-Level Matching Rule:
--------------------------
1. An alert matches a ground-truth fault event if:
   a. alert.asset_id == event.asset_id
   b. The alert temporal window [alert.event_start_estimate, alert.detected_at]
      overlaps with the event window [event.start_time - tolerance, event.end_time + tolerance].
2. Event-level deduplication: If multiple alerts match the same event, they count as
   ONE True Positive (TP) event match, avoiding artificial inflation.
3. False Positives (FP): Any generated alert that does not overlap with any ground-truth
   event on that asset is counted as an FP.
4. False Negatives (FN): Any ground-truth event with zero matching alerts is an FN.
5. Detection Delay: For each detected event, the elapsed time in minutes from event.start_time
   to the first matching alert's detected_at (clamped to >= 0.0 minutes).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campus_energy.analytics.anomalies import detect_anomalies
from campus_energy.analytics.features import engineer_features
from campus_energy.eval.baselines import detect_rolling_zscore, detect_static_threshold
from campus_energy.schemas import Alert

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvalResult:
    """Holds evaluation metrics for a single detector configuration."""

    model_name: str
    total_alerts: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    mean_delay_minutes: float
    median_delay_minutes: float
    per_fault_type: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_alerts": self.total_alerts,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "mean_delay_minutes": round(self.mean_delay_minutes, 2),
            "median_delay_minutes": round(self.median_delay_minutes, 2),
            "per_fault_type": self.per_fault_type,
        }


def load_and_prepare_data(
    data_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and timezone-normalize telemetry, context, assets, and ground truth."""
    if data_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        data_dir = base_dir / "data" / "generated"
    else:
        data_dir = Path(data_dir)

    tel_path = data_dir / "telemetry.csv"
    ctx_path = data_dir / "context.csv"
    assets_path = data_dir / "assets.csv"
    gt_path = data_dir / "anomaly_ground_truth.csv"

    if not all(p.exists() for p in (tel_path, ctx_path, assets_path, gt_path)):
        raise FileNotFoundError(f"Missing required data files in {data_dir}")

    tel = pd.read_csv(tel_path)
    ctx = pd.read_csv(ctx_path)
    assets = pd.read_csv(assets_path)
    gt = pd.read_csv(gt_path)

    # Standardize timestamps: strip timezones for consistent alignment
    tel["timestamp"] = pd.to_datetime(tel["timestamp"]).dt.tz_localize(None)
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"]).dt.tz_localize(None)
    gt["start_time"] = pd.to_datetime(gt["start_time"]).dt.tz_localize(None)
    gt["end_time"] = pd.to_datetime(gt["end_time"]).dt.tz_localize(None)

    # Attach spatial context to telemetry
    tel = tel.merge(assets[["asset_id", "building_id", "zone_id"]], on="asset_id", how="left")
    merged = tel.merge(ctx, on=["timestamp", "building_id", "zone_id"], how="left")

    features = engineer_features(merged, ctx)
    return features, assets, gt


def match_alerts_to_ground_truth(
    alerts: list[Alert],
    gt_df: pd.DataFrame,
    tolerance_minutes: int = 60,
    model_name: str = "Detector",
) -> AnomalyEvalResult:
    """Evaluate alerts against ground truth events with strict event-level accounting."""
    tolerance_delta = pd.Timedelta(minutes=tolerance_minutes)

    matched_event_ids: set[str] = set()
    event_delays: dict[str, float] = {}
    tp_alert_indices: set[int] = set()

    # Pre-parse alert windows
    parsed_alerts = []
    for i, a in enumerate(alerts):
        a_start = pd.to_datetime(a.event_start_estimate)
        if a_start.tzinfo is not None:
            a_start = a_start.tz_localize(None)
        a_end = pd.to_datetime(a.detected_at)
        if a_end.tzinfo is not None:
            a_end = a_end.tz_localize(None)
        parsed_alerts.append((i, a.asset_id, a_start, a_end, a.anomaly_type))

    # Evaluate each ground truth event
    for _, event_row in gt_df.iterrows():
        e_id = str(event_row["event_id"])
        e_asset = str(event_row["asset_id"])
        e_start = pd.to_datetime(event_row["start_time"])
        if e_start.tzinfo is not None:
            e_start = e_start.tz_localize(None)
        e_end = pd.to_datetime(event_row["end_time"])
        if e_end.tzinfo is not None:
            e_end = e_end.tz_localize(None)

        first_detection: pd.Timestamp | None = None

        for idx, a_asset, a_start, a_end, _ in parsed_alerts:
            if a_asset != e_asset:
                continue

            # Check overlap within tolerance window
            overlap = not (a_end < e_start - tolerance_delta or a_start > e_end + tolerance_delta)
            if overlap:
                matched_event_ids.add(e_id)
                tp_alert_indices.add(idx)
                if first_detection is None or a_end < first_detection:
                    first_detection = a_end

        if first_detection is not None:
            delay = (first_detection - e_start).total_seconds() / 60.0
            event_delays[e_id] = max(0.0, float(delay))

    total_gt = len(gt_df)
    tp = len(matched_event_ids)
    fn = total_gt - tp
    fp = len(parsed_alerts) - len(tp_alert_indices)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    delays = list(event_delays.values())
    mean_delay = float(np.mean(delays)) if delays else 0.0
    median_delay = float(np.median(delays)) if delays else 0.0

    # Per-fault-type breakdown
    per_fault_type: dict[str, dict[str, Any]] = {}
    for fault_type, group in gt_df.groupby("event_type"):
        g_ids = set(group["event_id"].astype(str))
        g_tp = len(g_ids.intersection(matched_event_ids))
        g_fn = len(g_ids) - g_tp
        g_rec = float(g_tp / len(g_ids)) if len(g_ids) > 0 else 0.0
        g_delays = [event_delays[eid] for eid in g_ids if eid in event_delays]
        per_fault_type[fault_type] = {
            "total_events": len(g_ids),
            "detected": g_tp,
            "missed": g_fn,
            "recall": round(g_rec, 4),
            "mean_delay_minutes": round(float(np.mean(g_delays)), 2) if g_delays else None,
        }

    return AnomalyEvalResult(
        model_name=model_name,
        total_alerts=len(alerts),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        mean_delay_minutes=mean_delay,
        median_delay_minutes=median_delay,
        per_fault_type=per_fault_type,
    )


def run_anomaly_evaluation(
    data_dir: Path | str | None = None,
    tolerance_minutes: int = 60,
) -> dict[str, Any]:
    """Execute complete anomaly detection evaluation across layered engine, ablated layers, and baselines."""
    features, assets, gt = load_and_prepare_data(data_dir)

    configurations = [
        ("Layered Engine (Full)", lambda: detect_anomalies(features, assets)),
        ("Ablation: Statistical Ridge", lambda: detect_anomalies(features, assets, enabled_layers=["statistical"])),
        ("Ablation: Unoccupied Rule", lambda: detect_anomalies(features, assets, enabled_layers=["unoccupied"])),
        ("Ablation: Hydraulic Dry-Run Rule", lambda: detect_anomalies(features, assets, enabled_layers=["pump_dry_run"])),
        ("Ablation: Sensor Dropout Rule", lambda: detect_anomalies(features, assets, enabled_layers=["sensor_dropout"])),
        ("Ablation: Isolation Forest ML", lambda: detect_anomalies(features, assets, enabled_layers=["isolation_forest"])),
        ("Baseline 1: Static Threshold", lambda: detect_static_threshold(features, assets, percentile=95.0, min_consecutive=3)),
        ("Baseline 2: Rolling Z-Score", lambda: detect_rolling_zscore(features, assets, window=288, z_threshold=3.0, min_consecutive=3)),
    ]

    results: list[dict[str, Any]] = []
    for name, run_fn in configurations:
        logger.info("Evaluating anomaly detector: %s", name)
        alerts = run_fn()
        eval_res = match_alerts_to_ground_truth(
            alerts=alerts,
            gt_df=gt,
            tolerance_minutes=tolerance_minutes,
            model_name=name,
        )
        results.append(eval_res.to_dict())

    return {
        "tolerance_minutes": tolerance_minutes,
        "total_ground_truth_events": len(gt),
        "results": results,
    }
