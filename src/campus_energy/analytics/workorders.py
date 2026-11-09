"""Fault Priority Queue and Maintenance Work-Order Generator.

Translates AI-detected telemetry anomalies, ground-truth events, and equipment
faults into an operational, prioritized CMMS-style work-order dispatch queue.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from campus_energy import assumptions


TEAM_MAPPING = {
    "hvac": "HVAC & Thermal Systems Team",
    "lighting": "Electrical & Lighting Maintenance",
    "pump": "Plumbing & Hydraulics Team",
    "bms": "BMS & IoT Automation Team",
    "general": "General Facility Maintenance",
}

SEVERITY_WEIGHTS: Dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
}

SLA_TARGET_HOURS: Dict[str, float] = {
    "critical": 4.0,
    "high": 12.0,
    "medium": 24.0,
    "low": 48.0,
}

DEFAULT_ACTIONS: Dict[str, str] = {
    "lights_left_on": "Audit room occupancy relays; ensure automated schedule setback and PIR sensor operation.",
    "hvac_left_on": "Inspect programmable thermostat setback; verify night-purge and motorized damper closure.",
    "pump_no_flow": "Emergency dispatch: inspect intake strainer, primer level, and pump cavitation protection.",
    "sensor_dropout": "Check physical cabling, edge gateway status, and IoT telemetry interface power.",
    "duplicate_reading": "Recalibrate edge telemetry node; verify broker message deduplication queue.",
    "comm_failure": "Restart field bus gateway; inspect serial RS-485/Modbus line termination.",
    "stale_sensor": "Inspect analog sensor probe for fouling or drift; trigger edge sensor diagnostic.",
    "corrupted_reading": "Verify ADC calibration and shielded cabling; check for EMI/RFI noise.",
    "power_spike": "Perform thermographic inspection on distribution panel; check for inrush surge or phase fault.",
    "gradual_increase": "Inspect motor bearings and compressor lubrication for mechanical wear degradation.",
    "overnight_excess": "Conduct off-hours sub-meter audit to isolate unmanaged phantom baseloads.",
    "occupancy_mismatch": "Synchronize BMS HVAC/lighting schedules with registrar class timetable.",
    "peer_anomaly": "Cross-compare submeter calibration against peer zones on same feeder floor.",
    "water_leak": "Acoustic leak inspection on riser pipeline; verify non-return valve and pressure zones.",
}

COLUMNS = [
    "wo_id",
    "event_id",
    "asset_id",
    "asset_name",
    "building_id",
    "event_type",
    "severity",
    "priority_score",
    "assigned_team",
    "sla_target",
    "sla_status",
    "est_waste_kwh_day",
    "est_waste_inr_day",
    "recommended_action",
    "status",
]


def determine_assigned_team(
    asset_type: str | None = None,
    event_type: str | None = None,
    asset_id: str | None = None,
) -> str:
    """Determine the responsible maintenance crew based on asset and fault characteristics."""
    ev_lower = str(event_type or "").lower()
    at_lower = str(asset_type or "").lower()
    aid_lower = str(asset_id or "").lower()

    # 1. BMS / IoT Automation Team
    if any(k in ev_lower for k in ["sensor", "dropout", "duplicate", "comm", "stale", "corrupt"]) or \
       any(k in at_lower for k in ["meter", "sensor", "telemetry"]) or \
       any(k in aid_lower for k in ["mtr-", "-mtr", "sensor"]):
        return TEAM_MAPPING["bms"]

    # 2. HVAC & Thermal Systems Team
    if any(k in ev_lower for k in ["hvac", "ac", "chiller", "cooling", "heating", "thermostat", "temperature"]) or \
       any(k in at_lower for k in ["hvac", "ac", "chiller", "cooling", "heat"]) or \
       any(k in aid_lower for k in ["-ac", "hvac", "chiller"]):
        return TEAM_MAPPING["hvac"]

    # 3. Electrical & Lighting Maintenance
    if any(k in ev_lower for k in ["light", "lighting", "illumination"]) or \
       any(k in at_lower for k in ["light", "lighting"]) or \
       any(k in aid_lower for k in ["-lt", "light"]):
        return TEAM_MAPPING["lighting"]

    # 4. Plumbing & Hydraulics Team
    if any(k in ev_lower for k in ["pump", "water", "flow", "leak", "pipe", "hydraulic", "pressure"]) or \
       any(k in at_lower for k in ["pump", "water", "plumb"]) or \
       any(k in aid_lower for k in ["-pm", "pump", "water"]):
        return TEAM_MAPPING["pump"]

    # 5. Default
    return TEAM_MAPPING["general"]


def calculate_sla_status(severity: str, hours_unresolved: float) -> tuple[str, str]:
    """Calculate SLA Target string and current status countdown badge."""
    sev_clean = str(severity).lower().strip()
    target_hours = SLA_TARGET_HOURS.get(sev_clean, 24.0)
    sla_target = f"{int(target_hours)} hours"

    remaining = target_hours - hours_unresolved
    if remaining > 0:
        if sev_clean == "critical":
            sla_status = f"🚨 P1 — {remaining:.1f}h Remaining"
        elif sev_clean == "high":
            sla_status = f"⚠️ P2 — {remaining:.1f}h Remaining"
        elif sev_clean == "medium":
            sla_status = f"🟡 P3 — {remaining:.1f}h Remaining"
        else:
            sla_status = f"🟢 P4 — {remaining:.1f}h Remaining"
    else:
        overdue = abs(remaining)
        sla_status = f"🔴 Overdue by {overdue:.1f}h"

    return sla_target, sla_status


def estimate_energy_waste(
    rated_power_kw: float | None,
    severity: str,
    event_type: str | None = None,
) -> tuple[float, float]:
    """Estimate daily waste in kWh and INR using assumptions registry tariff."""
    sev_clean = str(severity).lower().strip()

    if rated_power_kw is not None and not pd.isna(rated_power_kw) and float(rated_power_kw) > 0:
        power = float(rated_power_kw)
        # Factor representing typical daily excess run-hours (8 - 12 hrs)
        est_kwh = power * 10.0
    else:
        # Sensible defaults by severity
        defaults = {
            "critical": 120.0,
            "high": 80.0,
            "medium": 45.0,
            "low": 20.0,
        }
        est_kwh = defaults.get(sev_clean, 40.0)

    est_kwh = round(max(15.0, est_kwh), 1)
    cost_per_kwh = assumptions.get_value("tariff.electricity_commercial")
    est_inr = round(est_kwh * cost_per_kwh, 2)
    return est_kwh, est_inr


def calculate_priority_score(
    severity: str,
    est_waste_inr_day: float,
    hours_unresolved: float,
) -> float:
    """Compute normalized priority score (0-100).
    
    Formula:
      - Severity weight (40%): Critical=40, High=30, Medium=20, Low=10
      - Daily waste cost (40%): Capped at ₹1200/day
      - Hours elapsed/unresolved (20%): Urgency factor capped at 24 hours
    """
    sev_clean = str(severity).lower().strip()
    sev_weight = SEVERITY_WEIGHTS.get(sev_clean, 2)
    sev_comp = (sev_weight / 4.0) * 40.0

    waste_comp = min(max(0.0, est_waste_inr_day) / 1200.0, 1.0) * 40.0
    time_comp = min(max(0.0, hours_unresolved) / 24.0, 1.0) * 20.0

    score = round(sev_comp + waste_comp + time_comp, 1)
    return float(np.clip(score, 0.0, 100.0))


def create_work_orders(
    events_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    telemetry_df: pd.DataFrame | None = None,
    ref_time: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convert detected anomalies and events into a prioritized CMMS work-order queue.
    
    Parameters
    ----------
    events_df : pd.DataFrame
        Events, alerts, or anomaly detections.
    assets_df : pd.DataFrame
        Asset registry metadata.
    telemetry_df : pd.DataFrame, optional
        Telemetry data for reference timestamps.
    ref_time : datetime or pd.Timestamp, optional
        Reference cutoff time for SLA calculations.
        
    Returns
    -------
    pd.DataFrame
        Prioritized work orders sorted by priority_score descending.
    """
    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=COLUMNS)

    # Build quick lookup from assets_df
    assets_lookup: Dict[str, Dict[str, Any]] = {}
    if assets_df is not None and not assets_df.empty and "asset_id" in assets_df.columns:
        for _, row in assets_df.iterrows():
            aid = str(row["asset_id"])
            assets_lookup[aid] = row.to_dict()

    # Determine reference timestamp if not explicitly supplied
    if ref_time is None:
        if telemetry_df is not None and not telemetry_df.empty and "timestamp" in telemetry_df.columns:
            ref_time = pd.to_datetime(telemetry_df["timestamp"]).max()

    seen_wo_ids: set[str] = set()
    work_orders: List[Dict[str, Any]] = []

    for idx, row in events_df.iterrows():
        # Handle event/alert ID variations
        event_id = str(row.get("event_id") or row.get("alert_id") or f"EVT-{idx+1:04d}")
        
        # Unique Work Order ID: WO-{hash or 4-digit code}
        base_hash = hashlib.md5(event_id.encode("utf-8")).hexdigest()[:4].upper()
        wo_id = f"WO-{base_hash}"
        if wo_id in seen_wo_ids:
            wo_id = f"WO-{base_hash}-{idx+1}"
        seen_wo_ids.add(wo_id)

        asset_id = str(row.get("asset_id") or "UNKNOWN-ASSET")
        asset_info = assets_lookup.get(asset_id, {})

        # Asset metadata
        asset_name = str(asset_info.get("asset_name") or row.get("asset_name") or asset_id)
        building_id = str(asset_info.get("building_id") or row.get("building_id") or "BLD-MAIN")
        asset_type = str(asset_info.get("asset_type") or row.get("asset_type") or "equipment")
        rated_power_kw = asset_info.get("rated_power_kw")
        if rated_power_kw is None or pd.isna(rated_power_kw):
            rated_power_kw = row.get("rated_power_kw")

        event_type = str(row.get("event_type") or row.get("anomaly_type") or "general_fault")
        severity_val = row.get("severity", "medium")
        if hasattr(severity_val, "value"):
            severity = str(severity_val.value).lower()
        else:
            severity = str(severity_val).lower()

        # Assigned Team
        assigned_team = determine_assigned_team(
            asset_type=asset_type,
            event_type=event_type,
            asset_id=asset_id,
        )

        # Elapsed hours / unresolved duration
        hours_unresolved = 2.0
        if "hours_unresolved" in row and not pd.isna(row["hours_unresolved"]):
            hours_unresolved = float(row["hours_unresolved"])
        elif "start_time" in row and not pd.isna(row["start_time"]):
            st_dt = pd.to_datetime(row["start_time"])
            if "end_time" in row and not pd.isna(row["end_time"]):
                end_dt = pd.to_datetime(row["end_time"])
                dur = (end_dt - st_dt).total_seconds() / 3600.0
                hours_unresolved = max(0.1, dur)
            elif ref_time is not None:
                dur = (pd.to_datetime(ref_time) - st_dt).total_seconds() / 3600.0
                hours_unresolved = max(0.1, dur)
        elif "detected_at" in row and not pd.isna(row["detected_at"]):
            det_dt = pd.to_datetime(row["detected_at"])
            if ref_time is not None:
                dur = (pd.to_datetime(ref_time) - det_dt).total_seconds() / 3600.0
                hours_unresolved = max(0.1, dur)

        # Waste calculations
        est_kwh, est_inr = estimate_energy_waste(rated_power_kw, severity, event_type)

        # Priority score
        priority_score = calculate_priority_score(severity, est_inr, hours_unresolved)

        # SLA Target and Countdown Status
        sla_target, sla_status = calculate_sla_status(severity, hours_unresolved)

        # Actionable recommendation
        action = str(row.get("recommendation") or row.get("recommended_action") or "").strip()
        if not action or action.lower() == "none":
            action = DEFAULT_ACTIONS.get(event_type, "Dispatch field technician for visual inspection and operational audit.")

        status = str(row.get("status") or "Open")

        work_orders.append({
            "wo_id": wo_id,
            "event_id": event_id,
            "asset_id": asset_id,
            "asset_name": asset_name,
            "building_id": building_id,
            "event_type": event_type,
            "severity": severity,
            "priority_score": priority_score,
            "assigned_team": assigned_team,
            "sla_target": sla_target,
            "sla_status": sla_status,
            "est_waste_kwh_day": est_kwh,
            "est_waste_inr_day": est_inr,
            "recommended_action": action,
            "status": status,
        })

    result_df = pd.DataFrame(work_orders, columns=COLUMNS)
    result_df = result_df.sort_values(by="priority_score", ascending=False).reset_index(drop=True)
    return result_df


def get_work_order_summary(wo_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute executive and operational summary metrics for the work order queue."""
    if wo_df is None or wo_df.empty:
        return {
            "total_count": 0,
            "active_count": 0,
            "critical_count": 0,
            "daily_waste_kwh": 0.0,
            "daily_waste_inr": 0.0,
            "monthly_recovery_inr": 0.0,
            "top_priority_asset": "None",
            "top_priority_score": 0.0,
            "top_wo_id": "N/A",
        }

    active = wo_df[wo_df["status"] != "Resolved"]
    critical = active[active["severity"].str.lower().isin(["critical", "high"])]
    daily_kwh = float(active["est_waste_kwh_day"].sum())
    daily_inr = float(active["est_waste_inr_day"].sum())
    monthly_inr = round(daily_inr * 30.0, 2)

    top_asset = "None"
    top_score = 0.0
    top_wo = "N/A"
    if not active.empty:
        top_row = active.iloc[0]
        top_asset = str(top_row.get("asset_name") or top_row.get("asset_id"))
        top_score = float(top_row.get("priority_score", 0.0))
        top_wo = str(top_row.get("wo_id", "N/A"))

    return {
        "total_count": len(wo_df),
        "active_count": len(active),
        "critical_count": len(critical),
        "daily_waste_kwh": round(daily_kwh, 1),
        "daily_waste_inr": round(daily_inr, 2),
        "monthly_recovery_inr": monthly_inr,
        "top_priority_asset": top_asset,
        "top_priority_score": top_score,
        "top_wo_id": top_wo,
    }
