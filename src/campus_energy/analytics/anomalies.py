"""Anomaly detection module combining rules, statistical, and ML methods.

Detection layers (in order of sophistication):
  1. Rule-based: hard-coded physical rules (unoccupied-on, pump-no-flow, etc.)
  2. Statistical: deviation from a Ridge-regression baseline model
  3. ML: Isolation Forest for multivariate outlier detection
  4. Deduplication: merge overlapping alerts for the same asset/type
"""
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from campus_energy import assumptions
from campus_energy.schemas import Alert, Severity, AlertStatus
from campus_energy.analytics.baseline import ExpectedConsumptionModel


def _make_alert(
    asset_id: str,
    building_id: str,
    zone_id,
    anomaly_type: str,
    severity: Severity,
    evidence: str,
    recommendation: str,
    detected_at,
    event_start,
    actual_value: float = None,
    expected_value: float = None,
    deviation: float = None,
    confidence: float = None,
) -> Alert:
    return Alert(
        alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
        detected_at=detected_at,
        event_start_estimate=event_start,
        asset_id=asset_id,
        building_id=building_id,
        zone_id=zone_id,
        anomaly_type=anomaly_type,
        severity=severity,
        actual_value=round(actual_value, 2) if actual_value is not None else None,
        expected_value=round(expected_value, 2) if expected_value is not None else None,
        deviation=round(deviation, 2) if deviation is not None else None,
        evidence=evidence,
        recommendation=recommendation,
        confidence=confidence,
    )


def _deduplicate_alerts(alerts: list[Alert]) -> list[Alert]:
    """Merge alerts for the same (asset_id, anomaly_type) within deduplication window."""
    if not alerts:
        return alerts

    seen = {}  # (asset_id, anomaly_type) → latest Alert
    deduped = []
    dedup_hours = float(assumptions.get_value("anomaly.alert.dedup_window_hours"))

    for alert in sorted(alerts, key=lambda a: a.detected_at):
        key = (alert.asset_id, alert.anomaly_type)
        if key in seen:
            prev = seen[key]
            gap = (alert.detected_at - prev.detected_at).total_seconds() / 3600
            if gap <= dedup_hours:
                # Keep the more severe one (or latest)
                order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
                if order.index(alert.severity) >= order.index(prev.severity):
                    seen[key] = alert
                continue
        seen[key] = alert
        deduped.append(alert)

    return deduped


def detect_anomalies(features_df: pd.DataFrame, assets_df: pd.DataFrame) -> list[Alert]:
    """Run rules + statistical + ML detectors over the features dataframe.

    In a real system, this runs incrementally on each new batch.
    For the prototype, we run batch-wise over the full history.
    """
    alerts: list[Alert] = []

    min_time = features_df["timestamp"].min()
    train_end = min_time + pd.Timedelta(days=7)

    # Load anomaly parameters from registry
    dev_cutoff = float(assumptions.get_value("anomaly.statistical.deviation_cutoff_kw"))
    pct_cutoff = float(assumptions.get_value("anomaly.statistical.pct_deviation_cutoff"))
    crit_cutoff = float(assumptions.get_value("anomaly.statistical.critical_deviation_kw"))
    standby_fraction = float(assumptions.get_value("anomaly.unoccupied.standby_fraction"))
    pump_dry_fraction = float(assumptions.get_value("anomaly.pump.dry_run_power_fraction"))
    if_contamination = float(assumptions.get_value("anomaly.ml.isolation_forest_contamination"))
    ambient_fallback = float(assumptions.get_value("comfort.ambient_fallback_c"))

    asset_ids = features_df["asset_id"].unique()

    for asset_id in asset_ids:
        asset_info_rows = assets_df[assets_df["asset_id"] == asset_id]
        if asset_info_rows.empty:
            continue
        asset_info = asset_info_rows.iloc[0]
        building_id = asset_info["building_id"]
        zone_id = asset_info["zone_id"] if pd.notna(asset_info.get("zone_id")) else None

        asset_df = features_df[features_df["asset_id"] == asset_id].copy().sort_values("timestamp")
        # Ensure context columns exist with safe defaults (they may not be merged in
        # when engineer_features is called without context data)
        if "occupancy" not in asset_df.columns:
            asset_df["occupancy"] = 0
        if "outdoor_temperature_c" not in asset_df.columns:
            asset_df["outdoor_temperature_c"] = ambient_fallback
        train_df = asset_df[asset_df["timestamp"] <= train_end]
        test_df = asset_df[asset_df["timestamp"] > train_end]

        if len(train_df) < 50 or len(test_df) == 0:
            continue

        # ------------------------------------------------------------------ #
        # Layer 1 — Statistical baseline (Ridge regression)                  #
        # ------------------------------------------------------------------ #
        model = ExpectedConsumptionModel()
        model.fit(train_df)

        asset_df.loc[test_df.index, "expected_power_kw"] = model.predict(test_df)
        asset_df.loc[test_df.index, "deviation"] = (
            asset_df.loc[test_df.index, "power_kw"]
            - asset_df.loc[test_df.index, "expected_power_kw"]
        )

        # High deviation (actual >> expected)
        dev_mask = (
            (asset_df["timestamp"] > train_end)
            & (asset_df["deviation"] > dev_cutoff)
            & (asset_df["deviation"] > pct_cutoff * asset_df["expected_power_kw"].clip(lower=0.1))
        )
        asset_df["is_dev"] = dev_mask.astype(int)
        asset_df["dev_event"] = (asset_df["is_dev"].diff() == 1).cumsum()

        for _, group in asset_df[asset_df["is_dev"] == 1].groupby("dev_event"):
            if len(group) >= 6:  # ≥ 30 min sustained
                dev_mean = group["deviation"].mean()
                exp_mean = group["expected_power_kw"].mean()
                confidence = min(0.95, 0.5 + len(group) / 200)
                alerts.append(
                    _make_alert(
                        asset_id=asset_id,
                        building_id=building_id,
                        zone_id=zone_id,
                        anomaly_type="high_consumption",
                        severity=Severity.CRITICAL if dev_mean > crit_cutoff else Severity.MEDIUM,
                        evidence=f"Power exceeded baseline by {dev_mean:.2f} kW for {len(group)*5} min.",
                        recommendation="Inspect equipment for efficiency degradation or unintended operation.",
                        detected_at=group["timestamp"].max(),
                        event_start=group["timestamp"].min(),
                        actual_value=group["power_kw"].mean(),
                        expected_value=exp_mean,
                        deviation=dev_mean,
                        confidence=confidence,
                    )
                )

        # ------------------------------------------------------------------ #
        # Layer 2 — Rule: Equipment ON while zone is unoccupied              #
        # ------------------------------------------------------------------ #
        asset_type = asset_info.get("asset_type", "")
        rated_kw = asset_info.get("rated_power_kw", 1.0) or 1.0

        if asset_type in ["lighting", "hvac", "computer_lab"]:
            unocc_mask = (
                (asset_df["occupancy"] == 0)
                & (asset_df["power_kw"] > rated_kw * standby_fraction)
            )
            asset_df["is_unocc"] = unocc_mask.astype(int)
            asset_df["unocc_event"] = (asset_df["is_unocc"].diff() == 1).cumsum()

            for _, group in asset_df[asset_df["is_unocc"] == 1].groupby("unocc_event"):
                if len(group) >= 6:
                    alerts.append(
                        _make_alert(
                            asset_id=asset_id,
                            building_id=building_id,
                            zone_id=zone_id,
                            anomaly_type="unoccupied_operation",
                            severity=Severity.MEDIUM if asset_type == "lighting" else Severity.HIGH,
                            evidence=f"{asset_type.upper()} ran at {group['power_kw'].mean():.2f} kW with zero occupancy for {len(group)*5} min.",
                            recommendation="Enforce auto-off policy via BMS schedule.",
                            detected_at=group["timestamp"].max(),
                            event_start=group["timestamp"].min(),
                            actual_value=group["power_kw"].mean(),
                            expected_value=0.0,
                            confidence=0.9,
                        )
                    )

        # ------------------------------------------------------------------ #
        # Layer 3 — Rule: Pump running with no flow                          #
        # ------------------------------------------------------------------ #
        if asset_type == "pump" and "flow_lpm" in asset_df.columns:
            noflow_mask = (
                (asset_df["power_kw"] > rated_kw * pump_dry_fraction)
                & (asset_df["flow_lpm"] == 0)
            )
            asset_df["is_noflow"] = noflow_mask.astype(int)
            asset_df["noflow_event"] = (asset_df["is_noflow"].diff() == 1).cumsum()

            for _, group in asset_df[asset_df["is_noflow"] == 1].groupby("noflow_event"):
                if len(group) >= 3:
                    alerts.append(
                        _make_alert(
                            asset_id=asset_id,
                            building_id=building_id,
                            zone_id=zone_id,
                            anomaly_type="pump_no_flow",
                            severity=Severity.HIGH,
                            evidence="Pump drawing power but reporting 0 flow — possible blockage or dry-run.",
                            recommendation="Dispatch maintenance immediately. Risk of pump burnout.",
                            detected_at=group["timestamp"].max(),
                            event_start=group["timestamp"].min(),
                            actual_value=0.0,
                            expected_value=asset_info.get("flow_rated_lpm", 100.0),
                            confidence=0.95,
                        )
                    )

        # ------------------------------------------------------------------ #
        # Layer 4 — Rule: Sensor dropout                                     #
        # ------------------------------------------------------------------ #
        if "quality_flag" in asset_df.columns and "sensor_status" in asset_df.columns:
            dropout_mask = (
                (asset_df["quality_flag"] == "missing")
                | (asset_df["sensor_status"] == "offline")
            )
            asset_df["is_drop"] = dropout_mask.astype(int)
            asset_df["drop_event"] = (asset_df["is_drop"].diff() == 1).cumsum()

            for _, group in asset_df[asset_df["is_drop"] == 1].groupby("drop_event"):
                if len(group) >= 3:
                    alerts.append(
                        _make_alert(
                            asset_id=asset_id,
                            building_id=building_id,
                            zone_id=zone_id,
                            anomaly_type="sensor_dropout",
                            severity=Severity.LOW,
                            evidence=f"Sensor offline / missing data for {len(group)*5} min.",
                            recommendation="Check sensor connectivity and meter communication bus.",
                            detected_at=group["timestamp"].max(),
                            event_start=group["timestamp"].min(),
                            confidence=0.99,
                        )
                    )

        # ------------------------------------------------------------------ #
        # Layer 5 — ML: Isolation Forest (multivariate)                      #
        # ------------------------------------------------------------------ #
        features_for_ml = ["power_kw", "power_ma_3h", "power_std_3h", "hour"]
        ml_cols = [c for c in features_for_ml if c in test_df.columns]

        if len(ml_cols) >= 3 and len(test_df) >= 30:
            X = test_df[ml_cols].copy().fillna(0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            clf = IsolationForest(
                n_estimators=100,
                contamination=if_contamination,
                random_state=42,
            )
            preds = clf.fit_predict(X_scaled)
            scores = clf.score_samples(X_scaled)  # lower = more anomalous

            ml_outlier_df = test_df.copy()
            ml_outlier_df["if_label"] = preds
            ml_outlier_df["if_score"] = scores

            # Group contiguous outlier points
            ml_outlier_df["is_outlier"] = (ml_outlier_df["if_label"] == -1).astype(int)
            ml_outlier_df["outlier_event"] = (ml_outlier_df["is_outlier"].diff() == 1).cumsum()

            for _, group in ml_outlier_df[ml_outlier_df["is_outlier"] == 1].groupby("outlier_event"):
                if len(group) >= 4:  # At least 20 min
                    # Only flag if it's not already caught by rules
                    anomaly_score = -group["if_score"].mean()  # Normalize to positive
                    confidence = min(0.9, 0.5 + anomaly_score)
                    alerts.append(
                        _make_alert(
                            asset_id=asset_id,
                            building_id=building_id,
                            zone_id=zone_id,
                            anomaly_type="ml_multivariate_anomaly",
                            severity=Severity.MEDIUM if anomaly_score < 0.7 else Severity.HIGH,
                            evidence=(
                                f"Isolation Forest flagged {len(group)} anomalous readings "
                                f"(avg anomaly score: {anomaly_score:.3f}). "
                                f"Power: {group['power_kw'].mean():.2f} kW vs 3h avg {group['power_ma_3h'].mean():.2f} kW."
                            ),
                            recommendation="Investigate asset telemetry pattern; may indicate degradation or unusual load.",
                            detected_at=group["timestamp"].max(),
                            event_start=group["timestamp"].min(),
                            actual_value=group["power_kw"].mean(),
                            confidence=round(confidence, 2),
                        )
                    )

    # Deduplicate and return
    return _deduplicate_alerts(alerts)
