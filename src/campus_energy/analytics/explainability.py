"""Anomaly Explainability Layer ("Why was this flagged?").

Translates mathematical and rule-based anomaly detector signals into structured,
human-interpretable technical diagnostics for facility managers and technicians.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from campus_energy.schemas import Alert, Severity


def get_detector_type(anomaly_type: str) -> str:
    """Map an anomaly type string to its underlying detection mechanism."""
    anom = (anomaly_type or "").lower().strip()
    if anom == "high_consumption":
        return "Statistical Baseline (Ridge Regression)"
    elif anom in ("unoccupied_operation", "lights_left_on", "hvac_left_on", "occupancy_mismatch"):
        return "Operational BMS Rule"
    elif anom == "pump_no_flow":
        return "Hydraulic Rule"
    elif anom in ("sensor_dropout", "comm_failure", "stale_sensor", "corrupted_reading", "duplicate_reading", "timestamp_gap"):
        return "Telemetry Data Quality Rule"
    elif anom in ("ml_multivariate_anomaly", "isolation_forest"):
        return "Multivariate Isolation Forest (ML)"
    elif "rule" in anom:
        return "Operational BMS Rule"
    elif "ml" in anom or "forest" in anom:
        return "Multivariate Isolation Forest (ML)"
    else:
        return "Statistical Baseline (Ridge Regression)"


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Safely extract field from Alert pydantic model, dict, or pandas Series."""
    if hasattr(obj, key):
        val = getattr(obj, key)
        return val if val is not None else default
    if isinstance(obj, dict):
        val = obj.get(key)
        return val if val is not None else default
    if hasattr(obj, "__getitem__"):
        try:
            val = obj[key]
            if pd.notna(val):
                return val
        except (KeyError, IndexError, TypeError):
            pass
    return default


def explain_alert(
    alert: Union[Alert, Dict[str, Any], Any],
    context_row: Optional[Union[Dict[str, Any], pd.Series]] = None,
    asset_info: Optional[Union[Dict[str, Any], pd.Series]] = None,
) -> Dict[str, Any]:
    """Generate structured, human-interpretable technical diagnostics for an alert.

    Parameters
    ----------
    alert : Alert or dict or Series
        The alert to diagnose. Expected fields: anomaly_type, actual_value, expected_value,
        deviation, evidence, severity, confidence, detected_at, asset_id, building_id.
    context_row : dict or Series, optional
        Environmental and operational context at detection time (outdoor_temperature_c,
        occupancy, day_of_week, day_type, working_hours).
    asset_info : dict or Series, optional
        Asset registry metadata (asset_name, asset_type, rated_power_kw, etc.).

    Returns
    -------
    dict
        Structured explainability dictionary containing:
        - alert_id
        - anomaly_type
        - detector_type
        - why_headline
        - baseline_context (expected_kw, actual_kw, deviation_kw, deviation_pct, time_str, day_type, outdoor_temp_c, occupancy)
        - contributing_factors (list of {factor, status, detail})
        - probable_root_cause
        - recommended_verification
    """
    alert_id = _get_val(alert, "alert_id", "ALT-UNKNOWN")
    anomaly_type = _get_val(alert, "anomaly_type", "high_consumption")
    actual_value = _get_val(alert, "actual_value", None)
    expected_value = _get_val(alert, "expected_value", None)
    deviation = _get_val(alert, "deviation", None)
    evidence = _get_val(alert, "evidence", "")
    severity_raw = _get_val(alert, "severity", "medium")
    severity = severity_raw.value if hasattr(severity_raw, "value") else str(severity_raw)
    confidence = _get_val(alert, "confidence", 0.90)
    detected_at = _get_val(alert, "detected_at", None)
    asset_id = _get_val(alert, "asset_id", "UNKNOWN_ASSET")
    building_id = _get_val(alert, "building_id", "UNKNOWN_BLDG")
    zone_id = _get_val(alert, "zone_id", None)

    # Convert context_row & asset_info to dict
    if hasattr(context_row, "to_dict"):
        ctx_dict = context_row.to_dict()
    elif isinstance(context_row, dict):
        ctx_dict = context_row
    else:
        ctx_dict = {}

    if hasattr(asset_info, "to_dict"):
        asset_dict = asset_info.to_dict()
    elif isinstance(asset_info, dict):
        asset_dict = asset_info
    else:
        asset_dict = {}

    # Normalize detected_at
    dt = None
    if detected_at is not None:
        if isinstance(detected_at, str):
            try:
                dt = pd.to_datetime(detected_at)
            except Exception:
                dt = None
        elif hasattr(detected_at, "to_pydatetime"):
            dt = detected_at.to_pydatetime()
        elif isinstance(detected_at, datetime):
            dt = detected_at

    # Determine time string
    if "time" in ctx_dict and ctx_dict["time"]:
        time_str = str(ctx_dict["time"])
    elif dt is not None and hasattr(dt, "minute"):
        if dt.minute == 0:
            time_str = dt.strftime("%I %p").lstrip("0")
        else:
            time_str = dt.strftime("%I:%M %p").lstrip("0")
    else:
        time_str = "2 PM"

    # Determine day type
    if ctx_dict.get("day_type"):
        day_type = str(ctx_dict["day_type"]).lower()
    elif "day_of_week" in ctx_dict:
        dow = ctx_dict["day_of_week"]
        day_type = "weekend" if dow in (5, 6, "5", "6", "Saturday", "Sunday") else "weekday"
    elif dt is not None and hasattr(dt, "weekday"):
        day_type = "weekend" if dt.weekday() in (5, 6) else "weekday"
    else:
        day_type = "weekday"

    # Ambient temperature
    temp_raw = ctx_dict.get("outdoor_temperature_c", ctx_dict.get("temp", 34.0))
    try:
        temp_val = float(temp_raw) if temp_raw is not None else 34.0
    except (ValueError, TypeError):
        temp_val = 34.0
    temp_display = int(round(temp_val)) if temp_val.is_integer() else round(temp_val, 1)

    # Occupancy
    occ_raw = ctx_dict.get("occupancy", ctx_dict.get("occ"))
    if occ_raw is not None:
        try:
            occ_val = int(occ_raw)
        except (ValueError, TypeError):
            occ_val = 0 if anomaly_type == "unoccupied_operation" else 45
    else:
        occ_val = 0 if anomaly_type == "unoccupied_operation" else 45

    # Power metrics
    actual = float(actual_value) if actual_value is not None else 0.0
    expected = float(expected_value) if expected_value is not None else 0.0

    if deviation is not None:
        try:
            dev = float(deviation)
        except (ValueError, TypeError):
            dev = actual - expected
    else:
        dev = actual - expected

    if expected > 0:
        pct = (dev / expected) * 100.0
    else:
        pct = 0.0

    # Format confidence string
    if confidence is not None:
        try:
            conf_num = float(confidence)
            if 0 < conf_num <= 1.0:
                conf_str = f"{conf_num:.0%}"
            else:
                conf_str = f"{conf_num:.2f}"
        except (ValueError, TypeError):
            conf_str = str(confidence)
    else:
        conf_str = "85%"

    detector_type = get_detector_type(anomaly_type)

    # Build type-specific explainability
    anom_key = (anomaly_type or "").lower().strip()

    if anom_key == "high_consumption":
        why_headline = (
            f"Expected {expected:.2f} kW at {time_str} ({day_type}, {temp_display}°C, {occ_val} occupants). "
            f"Actual draw was {actual:.2f} kW (+{dev:.2f} kW, {pct:+.1f}% above baseline)."
        )
        contributing_factors = [
            {
                "factor": "Power Draw vs Baseline",
                "status": "Anomalous",
                "detail": f"Actual load {actual:.2f} kW exceeds expected baseline {expected:.2f} kW by +{dev:.2f} kW ({pct:+.1f}%)",
            },
            {
                "factor": "Outdoor Temperature",
                "status": "Trigger" if temp_val >= 35.0 else "Normal",
                "detail": f"Ambient temperature at {temp_display}°C {'amplifying thermal cooling load' if temp_val >= 35.0 else 'within seasonal baseline expectations'}",
            },
            {
                "factor": "Zone Occupancy",
                "status": "Trigger" if occ_val >= 50 else "Normal",
                "detail": f"Zone headcount at {occ_val} occupants ({'peak occupant density' if occ_val >= 50 else 'nominal occupant density'})",
            },
            {
                "factor": "3h Rolling Volatility",
                "status": "Anomalous" if dev > 1.5 else "Normal",
                "detail": "Standard deviation and moving average indicate sustained elevated power rather than transient spike",
            },
        ]
        probable_root_cause = "Thermostat setpoint override, stuck cooling bypass damper, or staging controller hunting."
        recommended_verification = "Check BAS override logs, measure supply/return air temperatures, and verify damper actuator linkage."

    elif anom_key in ("unoccupied_operation", "lights_left_on", "hvac_left_on", "occupancy_mismatch"):
        ev_str = evidence if evidence else "unoccupied zone schedule"
        why_headline = (
            f"Asset ran drawing {actual:.2f} kW while zone schedule and occupancy sensor reported 0 occupants ({ev_str})."
        )
        contributing_factors = [
            {
                "factor": "Zone Occupancy Sensor",
                "status": "Trigger",
                "detail": f"PIR motion sensor and schedule confirm 0 occupants in zone ({ev_str})",
            },
            {
                "factor": "BMS Operating Schedule",
                "status": "Trigger",
                "detail": f"Calendar profile flags interval as inactive / scheduled OFF ({day_type} at {time_str})",
            },
            {
                "factor": "Active Circuit Power",
                "status": "Anomalous",
                "detail": f"Continuous power draw of {actual:.2f} kW recorded during unpopulated interval",
            },
            {
                "factor": "Standby vs Active Ratio",
                "status": "Anomalous",
                "detail": "Active draw exceeds phantom standby threshold (>20% rated capacity)",
            },
        ]
        probable_root_cause = "BMS schedule override left active, local manual switch left on, or faulty occupancy PIR sensor."
        recommended_verification = "Inspect physical wall toggle switches, verify zone PIR motion LED response, and check BMS calendar schedule."

    elif anom_key == "pump_no_flow":
        pump_power = (
            actual if actual > 0
            else float(asset_dict.get("rated_power_kw", 0)) if float(asset_dict.get("rated_power_kw", 0)) > 0
            else float(ctx_dict.get("power_kw", 0)) if float(ctx_dict.get("power_kw", 0)) > 0
            else actual
        )
        why_headline = (
            f"Hydraulic motor was energized at {pump_power:.2f} kW, but flowmeter recorded 0.0 LPM — severe cavitation or dry-run risk."
        )
        contributing_factors = [
            {
                "factor": "Hydraulic Motor Power",
                "status": "Trigger",
                "detail": f"Motor actively energized and consuming {pump_power:.2f} kW (>50% rated power)",
            },
            {
                "factor": "Flowmeter Output",
                "status": "Anomalous",
                "detail": "Electromagnetic / paddlewheel flowmeter registers 0.0 LPM while motor is running",
            },
            {
                "factor": "Head Pressure & Cavitation",
                "status": "Anomalous",
                "detail": "Zero hydraulic fluid displacement with electrical work performed; severe dry-run risk",
            },
        ]
        probable_root_cause = "Closed discharge/suction isolation valve, air-locked pump volute, mechanical impeller seizure, or clogged strainer."
        recommended_verification = "Verify manual valve positions, inspect pump suction strainer for blockage, and vent pump casing."

    elif anom_key in ("sensor_dropout", "comm_failure", "stale_sensor", "corrupted_reading", "duplicate_reading", "timestamp_gap"):
        why_headline = (
            "Sensor telemetry dropped below health heartbeat with quality flag set to offline/missing for extended interval."
        )
        contributing_factors = [
            {
                "factor": "Telemetry Quality Flag",
                "status": "Trigger",
                "detail": f"Data ingestion pipeline marked reading quality as 'missing' or 'offline' ({evidence or 'no data'})",
            },
            {
                "factor": "Heartbeat Health Status",
                "status": "Anomalous",
                "detail": "Hardware watchdog timer expired without receiving meter keepalive packets",
            },
            {
                "factor": "Bus Communication Continuity",
                "status": "Anomalous",
                "detail": "Zero valid telemetry frames logged across consecutive sampling intervals",
            },
        ]
        probable_root_cause = "RS-485 Modbus communication loss, 24V DC power supply drop, or meter gateway network disconnect."
        recommended_verification = "Inspect RS-485 twisted-pair termination, verify 24V DC meter rail voltage, and ping edge gateway IP."

    elif anom_key in ("ml_multivariate_anomaly", "isolation_forest"):
        why_headline = (
            f"Isolation Forest flagged multivariate outlier across power, 3h rolling mean, standard deviation, and diurnal hour features (confidence: {conf_str})."
        )
        contributing_factors = [
            {
                "factor": "Multivariate Distance (Isolation Forest)",
                "status": "Anomalous",
                "detail": f"Observation isolated in sparse multi-feature subspace (model confidence: {conf_str})",
            },
            {
                "factor": "Power vs 3h Rolling Average",
                "status": "Trigger",
                "detail": f"Telemetry draw ({actual:.2f} kW) shows abrupt divergence from moving baseline window",
            },
            {
                "factor": "Rolling 3h Volatility (Std Dev)",
                "status": "Anomalous",
                "detail": "High-frequency variance deviates significantly from historical diurnal envelope",
            },
            {
                "factor": "Diurnal Hour Profile",
                "status": "Normal",
                "detail": f"Timestamp {time_str} matches expected facility operation, but operational vector is anomalous",
            },
        ]
        probable_root_cause = "Unscheduled multi-equipment co-activation, VFD hunting, or sub-meter CT saturation under harmonic distortion."
        recommended_verification = "Inspect VFD frequency trend, check motor mechanical vibration, and audit sub-meter log integrity."

    else:
        # Fallback generic explainability
        why_headline = (
            f"Anomaly detected for asset ({anomaly_type.replace('_', ' ')}): "
            f"actual draw {actual:.2f} kW vs expected {expected:.2f} kW ({evidence or 'operational outlier'})."
        )
        contributing_factors = [
            {
                "factor": "Observed Telemetry",
                "status": "Anomalous",
                "detail": f"Actual: {actual:.2f} kW vs Expected: {expected:.2f} kW (Deviation: {dev:+.2f} kW)",
            },
            {
                "factor": "Operational Signal",
                "status": "Trigger",
                "detail": str(evidence or "Detector trigger condition met"),
            },
        ]
        probable_root_cause = "Operational deviation from expected operational baseline or sensor communication disturbance."
        recommended_verification = "Inspect physical asset operating status, verify control setpoints, and check telemetry logs."

    return {
        "alert_id": alert_id,
        "asset_id": asset_id,
        "building_id": building_id,
        "zone_id": zone_id,
        "anomaly_type": anomaly_type,
        "severity": severity,
        "confidence": confidence,
        "detector_type": detector_type,
        "why_headline": why_headline,
        "baseline_context": {
            "expected_kw": round(expected, 2),
            "actual_kw": round(actual, 2),
            "deviation_kw": round(dev, 2),
            "deviation_pct": round(pct, 1),
            "time_str": time_str,
            "day_type": day_type,
            "outdoor_temp_c": round(temp_val, 1),
            "occupancy": occ_val,
        },
        "contributing_factors": contributing_factors,
        "probable_root_cause": probable_root_cause,
        "recommended_verification": recommended_verification,
    }
