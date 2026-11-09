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


def calculate_confidence_score(
    detector_type: str,
    has_layer_agreement: bool = False,
    deviation: float | None = None,
    baseline_power: float | None = None,
    rated_power: float | None = None,
) -> float:
    """Formal confidence scoring formulation for anomaly alerts.

    Formula:
        confidence = min(1.0, max(0.1, base_confidence + layer_agreement_bonus + residual_magnitude_score))

    Components:
        1. base_confidence: Calibrated prior confidence based on detector physics/methodology:
           - Physical rule detector (unoccupied, pump_dry_run, sensor_dropout) = 0.85
           - Statistical baseline model (Ridge regression expected load) = 0.65
           - Machine Learning (Isolation Forest multivariate) = 0.50
        2. layer_agreement_bonus: +0.15 if multiple distinct detector layers flagged an anomaly
           on the same asset within an overlapping time window; 0.0 otherwise.
        3. residual_magnitude_score: Scaled by deviation magnitude relative to baseline/rated power:
           relative_dev = deviation / max(baseline_power or rated_power or 10.0, 0.1)
           residual_magnitude_score = min(0.15, max(0.0, relative_dev * 0.08))

    Parameters:
        detector_type: 'rule', 'statistical', or 'isolation_forest' / 'ml'.
        has_layer_agreement: True if >=2 layers independently confirmed anomaly.
        deviation: Magnitude of power deviation in kW (|actual - expected|).
        baseline_power: Expected or model baseline power in kW.
        rated_power: Equipment nameplate rated power in kW.

    Returns:
        Confidence score strictly bounded within [0.10, 1.00], rounded to 2 decimal places.
    """
    det_lower = str(detector_type or "").lower().strip()
    if any(k in det_lower for k in ["rule", "unoccupied", "pump_dry_run", "pump_no_flow", "sensor_dropout", "physical"]):
        base_confidence = 0.85
    elif any(k in det_lower for k in ["statistical", "high_consumption", "ridge"]):
        base_confidence = 0.65
    elif any(k in det_lower for k in ["isolation_forest", "ml", "multivariate"]):
        base_confidence = 0.50
    else:
        base_confidence = 0.60

    layer_agreement_bonus = 0.15 if has_layer_agreement else 0.0

    residual_magnitude_score = 0.0
    if deviation is not None and deviation > 0:
        ref_kw = (
            float(baseline_power)
            if baseline_power is not None and float(baseline_power) > 0.1
            else (float(rated_power) if rated_power is not None and float(rated_power) > 0.1 else 10.0)
        )
        rel_dev = deviation / max(ref_kw, 0.1)
        residual_magnitude_score = min(0.15, max(0.0, rel_dev * 0.08))

    raw_conf = base_confidence + layer_agreement_bonus + residual_magnitude_score
    return round(float(np.clip(raw_conf, 0.10, 1.00)), 2)


def determine_alert_severity(
    anomaly_type: str,
    deviation: float | None = None,
    crit_cutoff_kw: float | None = None,
    rated_power_kw: float | None = None,
    asset_type: str | None = None,
) -> Severity:
    """Implement cost- and operational impact-based severity grading.

    Grading Criteria:
    - Critical: deviation >= critical threshold (from assumptions registry for statistical high consumption)
                OR immediate safety/operational impact.
    - High: large excess kW or after-hours running of high-draw HVAC/pumps.
    - Medium: moderate excess power or prolonged lighting left on.
    - Low: minor sensor jitter, low-draw deviation, or sensor dropout/comm failure.
    """
    if crit_cutoff_kw is None:
        try:
            crit_cutoff_kw = float(assumptions.get_value("anomaly.statistical.critical_deviation_kw"))
        except Exception:
            crit_cutoff_kw = 50.0

    anom = str(anomaly_type or "").lower().strip()
    atype = str(asset_type or "").lower().strip()
    dev = float(deviation) if deviation is not None and not pd.isna(deviation) else 0.0

    # Low severity: minor sensor jitter / dropout / comm failure
    if anom in ("sensor_dropout", "duplicate_reading", "stale_sensor", "comm_failure"):
        return Severity.LOW

    # High severity: pump dry-run or after-hours running of high-draw HVAC
    if anom in ("pump_no_flow", "pump_dry_run"):
        return Severity.HIGH

    if anom in ("unoccupied_operation", "hvac_left_on"):
        if atype == "lighting" or anom == "lights_left_on":
            return Severity.MEDIUM
        return Severity.HIGH

    if anom in ("lights_left_on", "occupancy_mismatch"):
        return Severity.MEDIUM

    # Statistical / high_consumption:
    # Critical if deviation >= critical threshold (from assumptions)
    if anom in ("high_consumption", "power_spike", "statistical"):
        if dev >= crit_cutoff_kw and crit_cutoff_kw > 0:
            return Severity.CRITICAL
        elif dev >= (crit_cutoff_kw * 0.5):
            return Severity.HIGH
        return Severity.MEDIUM

    if anom in ("ml_multivariate_anomaly", "isolation_forest"):
        if dev >= crit_cutoff_kw and crit_cutoff_kw > 0:
            return Severity.CRITICAL
        return Severity.HIGH

    # Fallback based on dev
    if dev >= crit_cutoff_kw and crit_cutoff_kw > 0:
        return Severity.CRITICAL
    elif dev >= 10.0:
        return Severity.HIGH
    elif dev >= 5.0:
        return Severity.MEDIUM
    return Severity.LOW


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
    # Ensure confidence is calibrated and bounded
    if confidence is None:
        confidence = calculate_confidence_score(
            detector_type=anomaly_type,
            deviation=deviation,
            baseline_power=expected_value,
        )
    else:
        confidence = round(float(np.clip(confidence, 0.10, 1.00)), 2)

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


ALL_LAYERS = {"statistical", "unoccupied", "pump_dry_run", "sensor_dropout", "isolation_forest"}


def detect_anomalies(
    features_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    enabled_layers: list[str] | set[str] | None = None,
) -> list[Alert]:
    """Run rules + statistical + ML detectors over the features dataframe.

    In a real system, this runs incrementally on each new batch.
    For the prototype, we run batch-wise over the full history.

    Args:
        features_df: Telemetry and context feature dataset.
        assets_df: Asset registry dataframe.
        enabled_layers: Optional collection of layer names to execute.
            Valid options: 'statistical', 'unoccupied', 'pump_dry_run',
            'sensor_dropout', 'isolation_forest'. If None, all layers run.
    """
    alerts: list[Alert] = []
    candidates: list[dict] = []
    active_layers = set(enabled_layers) if enabled_layers is not None else ALL_LAYERS

    min_time = pd.to_datetime(features_df["timestamp"].min())
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
        asset_df["occupancy"] = asset_df["occupancy"].fillna(0)
        asset_df["outdoor_temperature_c"] = asset_df["outdoor_temperature_c"].fillna(ambient_fallback)

        # Standardize timestamps for time comparisons
        asset_ts = pd.to_datetime(asset_df["timestamp"])
        if min_time.tzinfo is not None and asset_ts.dt.tz is None:
            asset_ts = asset_ts.dt.tz_localize(min_time.tzinfo)
        elif min_time.tzinfo is None and asset_ts.dt.tz is not None:
            asset_ts = asset_ts.dt.tz_localize(None)

        train_mask = asset_ts <= train_end
        train_df = asset_df[train_mask]
        test_df = asset_df[~train_mask]

        if len(train_df) < 50 or len(test_df) == 0:
            continue

        # ------------------------------------------------------------------ #
        # Layer 1 — Statistical baseline (Ridge regression)                  #
        # ------------------------------------------------------------------ #
        if "statistical" in active_layers:
            model = ExpectedConsumptionModel()
            model.fit(train_df)

            asset_df.loc[test_df.index, "expected_power_kw"] = model.predict(test_df)
            asset_df.loc[test_df.index, "deviation"] = (
                asset_df.loc[test_df.index, "power_kw"]
                - asset_df.loc[test_df.index, "expected_power_kw"]
            )

            # High deviation (actual >> expected)
            dev_mask = (
                ~train_mask
                & (asset_df["deviation"] > dev_cutoff)
                & (asset_df["deviation"] > pct_cutoff * asset_df["expected_power_kw"].clip(lower=0.1))
            )
            asset_df["is_dev"] = dev_mask.astype(int)
            asset_df["dev_event"] = (asset_df["is_dev"].diff() == 1).cumsum()

            for _, group in asset_df[asset_df["is_dev"] == 1].groupby("dev_event"):
                if len(group) >= 6:  # ≥ 30 min sustained
                    dev_mean = float(group["deviation"].mean())
                    exp_mean = float(group["expected_power_kw"].mean())
                    candidates.append({
                        "asset_id": asset_id,
                        "building_id": building_id,
                        "zone_id": zone_id,
                        "anomaly_type": "high_consumption",
                        "detector_type": "statistical",
                        "layer_name": "statistical",
                        "detected_at": group["timestamp"].max(),
                        "event_start": group["timestamp"].min(),
                        "actual_value": float(group["power_kw"].mean()),
                        "expected_value": exp_mean,
                        "deviation": dev_mean,
                        "rated_power_kw": asset_info.get("rated_power_kw"),
                        "asset_type": asset_info.get("asset_type", "meter"),
                        "evidence": f"Power exceeded baseline by {dev_mean:.2f} kW for {len(group)*5} min.",
                        "recommendation": "Inspect equipment for efficiency degradation or unintended operation.",
                    })

        # ------------------------------------------------------------------ #
        # Layer 2 — Rule: Equipment ON while zone is unoccupied              #
        # ------------------------------------------------------------------ #
        if "unoccupied" in active_layers:
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
                        act_kw = float(group["power_kw"].mean())
                        candidates.append({
                            "asset_id": asset_id,
                            "building_id": building_id,
                            "zone_id": zone_id,
                            "anomaly_type": "unoccupied_operation",
                            "detector_type": "rule",
                            "layer_name": "unoccupied",
                            "detected_at": group["timestamp"].max(),
                            "event_start": group["timestamp"].min(),
                            "actual_value": act_kw,
                            "expected_value": 0.0,
                            "deviation": act_kw,
                            "rated_power_kw": rated_kw,
                            "asset_type": asset_type,
                            "evidence": f"{asset_type.upper()} ran at {act_kw:.2f} kW with zero occupancy for {len(group)*5} min.",
                            "recommendation": "Enforce auto-off policy via BMS schedule.",
                        })

        # ------------------------------------------------------------------ #
        # Layer 3 — Rule: Pump running with no flow                          #
        # ------------------------------------------------------------------ #
        if "pump_dry_run" in active_layers:
            asset_type = asset_info.get("asset_type", "")
            rated_kw = asset_info.get("rated_power_kw", 1.0) or 1.0
            if asset_type == "pump" and "flow_lpm" in asset_df.columns:
                noflow_mask = (
                    (asset_df["power_kw"] > rated_kw * pump_dry_fraction)
                    & (asset_df["flow_lpm"] == 0)
                )
                asset_df["is_noflow"] = noflow_mask.astype(int)
                asset_df["noflow_event"] = (asset_df["is_noflow"].diff() == 1).cumsum()

                for _, group in asset_df[asset_df["is_noflow"] == 1].groupby("noflow_event"):
                    if len(group) >= 3:
                        act_pwr = float(group["power_kw"].mean())
                        candidates.append({
                            "asset_id": asset_id,
                            "building_id": building_id,
                            "zone_id": zone_id,
                            "anomaly_type": "pump_no_flow",
                            "detector_type": "rule",
                            "layer_name": "pump_dry_run",
                            "detected_at": group["timestamp"].max(),
                            "event_start": group["timestamp"].min(),
                            "actual_value": 0.0,
                            "expected_value": float(asset_info.get("flow_rated_lpm", 100.0)),
                            "deviation": act_pwr,
                            "rated_power_kw": rated_kw,
                            "asset_type": asset_type,
                            "evidence": "Pump drawing power but reporting 0 flow — possible blockage or dry-run.",
                            "recommendation": "Dispatch maintenance immediately. Risk of pump burnout.",
                        })

        # ------------------------------------------------------------------ #
        # Layer 4 — Rule: Sensor dropout                                     #
        # ------------------------------------------------------------------ #
        if "sensor_dropout" in active_layers:
            if "quality_flag" in asset_df.columns and "sensor_status" in asset_df.columns:
                dropout_mask = (
                    (asset_df["quality_flag"] == "missing")
                    | (asset_df["sensor_status"] == "offline")
                )
                asset_df["is_drop"] = dropout_mask.astype(int)
                asset_df["drop_event"] = (asset_df["is_drop"].diff() == 1).cumsum()

                for _, group in asset_df[asset_df["is_drop"] == 1].groupby("drop_event"):
                    if len(group) >= 3:
                        candidates.append({
                            "asset_id": asset_id,
                            "building_id": building_id,
                            "zone_id": zone_id,
                            "anomaly_type": "sensor_dropout",
                            "detector_type": "rule",
                            "layer_name": "sensor_dropout",
                            "detected_at": group["timestamp"].max(),
                            "event_start": group["timestamp"].min(),
                            "actual_value": None,
                            "expected_value": None,
                            "deviation": None,
                            "rated_power_kw": asset_info.get("rated_power_kw"),
                            "asset_type": asset_info.get("asset_type"),
                            "evidence": f"Sensor offline / missing data for {len(group)*5} min.",
                            "recommendation": "Check sensor connectivity and meter communication bus.",
                        })

        # ------------------------------------------------------------------ #
        # Layer 5 — ML: Isolation Forest (multivariate)                      #
        # ------------------------------------------------------------------ #
        if "isolation_forest" in active_layers:
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
                        anomaly_score = -group["if_score"].mean()
                        act_kw = float(group["power_kw"].mean())
                        avg_3h = float(group["power_ma_3h"].mean())
                        candidates.append({
                            "asset_id": asset_id,
                            "building_id": building_id,
                            "zone_id": zone_id,
                            "anomaly_type": "ml_multivariate_anomaly",
                            "detector_type": "ml",
                            "layer_name": "isolation_forest",
                            "detected_at": group["timestamp"].max(),
                            "event_start": group["timestamp"].min(),
                            "actual_value": act_kw,
                            "expected_value": avg_3h,
                            "deviation": abs(act_kw - avg_3h),
                            "rated_power_kw": asset_info.get("rated_power_kw"),
                            "asset_type": asset_info.get("asset_type"),
                            "evidence": (
                                f"Isolation Forest flagged {len(group)} anomalous readings "
                                f"(avg anomaly score: {anomaly_score:.3f}). "
                                f"Power: {act_kw:.2f} kW vs 3h avg {avg_3h:.2f} kW."
                            ),
                            "recommendation": "Investigate asset telemetry pattern; may indicate degradation or unusual load.",
                        })

    # Calibrate confidence scores and assign impact-based severity across candidates
    for i, cand in enumerate(candidates):
        # Multi-layer agreement check: another candidate on same asset with overlapping window
        has_agreement = False
        agreeing_layer = None
        for j, other in enumerate(candidates):
            if i == j:
                continue
            if cand["asset_id"] == other["asset_id"] and cand["layer_name"] != other["layer_name"]:
                # Check temporal overlap (+/- 60 min buffer)
                overlap_start = max(cand["event_start"], other["event_start"])
                overlap_end = min(cand["detected_at"], other["detected_at"]) + pd.Timedelta(minutes=60)
                if overlap_start <= overlap_end:
                    has_agreement = True
                    agreeing_layer = other["layer_name"]
                    break

        conf = calculate_confidence_score(
            detector_type=cand["detector_type"],
            has_layer_agreement=has_agreement,
            deviation=cand["deviation"],
            baseline_power=cand["expected_value"],
            rated_power=cand["rated_power_kw"],
        )

        sev = determine_alert_severity(
            anomaly_type=cand["anomaly_type"],
            deviation=cand["deviation"],
            crit_cutoff_kw=crit_cutoff,
            rated_power_kw=cand["rated_power_kw"],
            asset_type=cand["asset_type"],
        )

        evidence = cand["evidence"]
        if has_agreement and agreeing_layer:
            evidence = f"{evidence} (Confirmed by multi-layer agreement with {agreeing_layer})"

        alerts.append(
            _make_alert(
                asset_id=cand["asset_id"],
                building_id=cand["building_id"],
                zone_id=cand["zone_id"],
                anomaly_type=cand["anomaly_type"],
                severity=sev,
                evidence=evidence,
                recommendation=cand["recommendation"],
                detected_at=cand["detected_at"],
                event_start=cand["event_start"],
                actual_value=cand["actual_value"],
                expected_value=cand["expected_value"],
                deviation=cand["deviation"],
                confidence=conf,
            )
        )

    # Deduplicate and return
    return _deduplicate_alerts(alerts)
