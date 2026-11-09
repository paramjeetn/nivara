"""Fault Injection Engine for Nivara Live Demonstrations.

Supports 5 canonical physical, electrical, hydraulic, and sensor fault scenarios:
1. `lights_left_on`: Classroom/office lights forced ON at rated capacity during night/off hours.
2. `pump_no_flow`: Pump running with zero flow / cavitation.
3. `pipe_leak_overnight`: Sudden water flow surge of +35 LPM during 01:00-05:00 AM.
4. `sensor_dropout`: Sensor fails or drops to 0 / missing quality flag.
5. `pf_drop`: Power factor drops to 0.72, breaching 0.90 OERC regulatory threshold.

All injected faults are recorded into an ActiveFaultRegistry with original slices preserved
for verification, closed-loop resolution, and counterfactual M&V verification.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from campus_energy.schemas import (
    EquipmentState,
    Event,
    EventType,
    QualityFlag,
    SensorStatus,
    Severity,
)

logger = logging.getLogger(__name__)

FAULT_TYPES: list[str] = [
    "lights_left_on",
    "pump_no_flow",
    "pipe_leak_overnight",
    "sensor_dropout",
    "pf_drop",
]


class ActiveFaultRegistry:
    """Thread-safe active registry tracking injected faults and pre-fault states."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._faults: dict[str, dict[str, Any]] = {}

    def register_fault(
        self,
        event: Event,
        original_slice: pd.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._faults[event.event_id] = {
                "event": event,
                "original_slice": original_slice.copy() if original_slice is not None else None,
                "injected_at": datetime.now(timezone.utc),
                "status": "active",
                "metadata": metadata or {},
            }

    def get_active_faults(self) -> list[Event]:
        with self._lock:
            return [
                v["event"]
                for v in self._faults.values()
                if v.get("status") == "active"
            ]

    def get_fault_record(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._faults.get(event_id)
            return dict(rec) if rec else None

    def get_fault(self, event_id: str) -> Event | None:
        with self._lock:
            rec = self._faults.get(event_id)
            return rec["event"] if rec else None

    def resolve_fault(self, event_id: str) -> bool:
        with self._lock:
            if event_id in self._faults:
                self._faults[event_id]["status"] = "resolved"
                self._faults[event_id]["resolved_at"] = datetime.now(timezone.utc)
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._faults.clear()

    def count(self) -> int:
        with self._lock:
            return sum(1 for v in self._faults.values() if v.get("status") == "active")


active_fault_registry = ActiveFaultRegistry()


def get_supported_fault_types() -> list[str]:
    """Return the 5 canonical fault injection types."""
    return list(FAULT_TYPES)


def get_fault_presets() -> dict[str, dict[str, Any]]:
    """Return presentation-ready preset configurations for the 5 canonical faults."""
    return {
        "lights_left_on": {
            "title": "Overnight Lighting Left ON",
            "icon": "💡",
            "category": "Operational Waste",
            "asset_hint": "lighting",
            "default_duration_hours": 4.0,
            "severity": Severity.MEDIUM.value,
            "description": "Classroom/office lighting left running at rated capacity overnight with zero room occupancy.",
            "expected_detection": "Physical Rule (Unoccupied Operation) + Ridge Statistical Baseline",
        },
        "pump_no_flow": {
            "title": "Pump Dry-Run / Cavitation",
            "icon": "🌊",
            "category": "Equipment Protection",
            "asset_hint": "pump",
            "default_duration_hours": 2.0,
            "severity": Severity.HIGH.value,
            "description": "Circulation pump running at full electrical load while reporting 0.0 LPM flow rate.",
            "expected_detection": "Physical Dry-Run Interlock Rule (P_elec > standby & Flow = 0)",
        },
        "pipe_leak_overnight": {
            "title": "Midnight Distribution Pipe Rupture",
            "icon": "💧",
            "category": "Water & Resource Loss",
            "asset_hint": "meter",
            "default_duration_hours": 4.0,
            "severity": Severity.HIGH.value,
            "description": "Sudden water flow rate surge of +35.0 LPM occurring during off-duty hours (01:00-05:00 AM).",
            "expected_detection": "Minimum Night Flow (MNF) Multiplier Baseline > 3x Normal",
        },
        "sensor_dropout": {
            "title": "IoT Edge Sensor Dropout",
            "icon": "📡",
            "category": "Telemetry & Comm Failure",
            "asset_hint": "meter",
            "default_duration_hours": 3.0,
            "severity": Severity.LOW.value,
            "description": "Field bus transmitter stops reporting data; telemetry payload flagged as missing / offline.",
            "expected_detection": "Telemetry Ingestion Data Quality Monitor (QualityFlag = missing)",
        },
        "pf_drop": {
            "title": "Power Factor Regulatory Breach",
            "icon": "⚡",
            "category": "Tariff & Regulatory Penalty",
            "asset_hint": "meter",
            "default_duration_hours": 4.0,
            "severity": Severity.HIGH.value,
            "description": "Substation capacitor bank trips, dropping power factor to 0.72 and violating 0.90 OERC threshold.",
            "expected_detection": "Indian Tariff Compliance Engine (OERC PF Surcharge Rule)",
        },
    }


def _normalize_timestamp(val: Any) -> pd.Timestamp:
    """Helper to convert string/datetime to a standardized pandas Timestamp."""
    ts = pd.to_datetime(val)
    if isinstance(ts, pd.Series):
        return ts
    return ts


def inject_live_fault(
    telemetry_df: pd.DataFrame,
    fault_type: str,
    asset_id: str,
    start_time: datetime | str | None = None,
    duration_hours: float = 4.0,
    **kwargs: Any,
) -> tuple[pd.DataFrame, Event]:
    """Inject a live operational or physical fault into a telemetry DataFrame.

    Parameters:
        telemetry_df: Live or historical telemetry DataFrame.
        fault_type: One of 'lights_left_on', 'pump_no_flow', 'pipe_leak_overnight',
            'sensor_dropout', or 'pf_drop'.
        asset_id: Target asset identifier.
        start_time: Injection start timestamp. If None, targets the most recent
            window of the telemetry dataset.
        duration_hours: Duration of the anomalous event in hours (default: 4.0).
        **kwargs: Optional overrides:
            - rated_power_kw: float (for lighting or pump)
            - surge_lpm: float (for pipe leak, default: 35.0)
            - power_factor: float (for pf_drop, default: 0.72)

    Returns:
        tuple[pd.DataFrame, Event]: (modified_telemetry_df, injected_event_record)
    """
    clean_type = str(fault_type or "").strip().lower()
    # Normalize aliases
    if clean_type in ("lights", "light", "lights_on", "unoccupied_lighting"):
        clean_type = "lights_left_on"
    elif clean_type in ("pump", "cavitation", "dry_run", "pump_dry"):
        clean_type = "pump_no_flow"
    elif clean_type in ("leak", "water_leak", "pipe_leak", "burst"):
        clean_type = "pipe_leak_overnight"
    elif clean_type in ("dropout", "sensor", "sensor_fail", "comm_fail", "comm_failure"):
        clean_type = "sensor_dropout"
    elif clean_type in ("pf", "power_factor", "low_pf", "power_factor_drop"):
        clean_type = "pf_drop"

    if clean_type not in FAULT_TYPES:
        raise ValueError(f"Unsupported fault type '{fault_type}'. Must be one of: {FAULT_TYPES}")

    tel = telemetry_df.copy()
    if "timestamp" not in tel.columns:
        raise ValueError("telemetry_df must contain a 'timestamp' column.")

    # Parse timestamps for alignment
    parsed_ts = pd.to_datetime(tel["timestamp"])
    is_tz_aware = parsed_ts.dt.tz is not None

    # Determine start and end time windows
    asset_mask_all = tel["asset_id"] == asset_id
    if start_time is None:
        if asset_mask_all.any():
            latest_asset_ts = parsed_ts[asset_mask_all].max()
            resolved_start = latest_asset_ts - pd.Timedelta(hours=duration_hours)
        else:
            latest_all_ts = parsed_ts.max() if not parsed_ts.empty else pd.Timestamp.now(timezone.utc)
            resolved_start = latest_all_ts - pd.Timedelta(hours=duration_hours)
    else:
        resolved_start = pd.to_datetime(start_time)
        if is_tz_aware and resolved_start.tzinfo is None:
            resolved_start = resolved_start.tz_localize(parsed_ts.dt.tz)
        elif not is_tz_aware and resolved_start.tzinfo is not None:
            resolved_start = resolved_start.tz_convert(None)

    resolved_end = resolved_start + pd.Timedelta(hours=duration_hours)

    # Filter mask for the target asset within the target window
    mask = (
        asset_mask_all
        & (parsed_ts >= resolved_start)
        & (parsed_ts <= resolved_end)
    )

    # Fallback: if time filter matched 0 rows but asset exists in telemetry,
    # target the latest available rows for that asset
    if not mask.any() and asset_mask_all.any():
        num_target_rows = max(1, int(duration_hours * 12))  # 5-min intervals
        sub_indices = tel[asset_mask_all].tail(num_target_rows).index
        mask = tel.index.isin(sub_indices)
        resolved_start = parsed_ts.loc[sub_indices].min()
        resolved_end = parsed_ts.loc[sub_indices].max()

    # If the asset does not exist at all in telemetry, generate synthetic baseline rows
    if not mask.any():
        logger.info("Asset %s not found in telemetry; synthesizing fault rows.", asset_id)
        step_delta = timedelta(minutes=5)
        num_steps = max(1, int(duration_hours * 12))
        synthetic_rows = []
        cur_t = resolved_start
        for i in range(num_steps):
            synthetic_rows.append({
                "reading_id": f"SYN-{asset_id}-{i:04d}",
                "timestamp": cur_t.isoformat() if hasattr(cur_t, "isoformat") else str(cur_t),
                "asset_id": asset_id,
                "reading_type": "meter" if clean_type in ("pf_drop", "sensor_dropout") else "equipment",
                "power_kw": 10.0,
                "interval_energy_kwh": 10.0 / 12.0,
                "cumulative_energy_kwh": 100.0 + (i * (10.0 / 12.0)),
                "voltage_v": 230.0,
                "current_a": 43.5,
                "power_factor": 0.95,
                "state": "ON",
                "flow_lpm": 50.0 if clean_type == "pump_no_flow" else 0.0,
                "temperature_c": 24.0,
                "setpoint_c": 24.0,
                "sensor_status": "healthy",
                "quality_flag": "valid",
            })
            cur_t += step_delta

        syn_df = pd.DataFrame(synthetic_rows)
        tel = pd.concat([tel, syn_df], ignore_index=True)
        parsed_ts = pd.to_datetime(tel["timestamp"])
        mask = tel["asset_id"] == asset_id
        resolved_start = parsed_ts[mask].min()
        resolved_end = parsed_ts[mask].max()

    original_slice = tel.loc[mask].copy()

    # Apply canonical fault mutations
    event_id = f"EVT-LIVE-{uuid.uuid4().hex[:6].upper()}"

    if clean_type == "lights_left_on":
        rated_power_kw = float(kwargs.get("rated_power_kw", 15.0))
        forced_kw = round(rated_power_kw * 0.95, 2)
        tel.loc[mask, "power_kw"] = forced_kw
        tel.loc[mask, "interval_energy_kwh"] = round(forced_kw / 12.0, 4)
        tel.loc[mask, "state"] = EquipmentState.ON.value
        tel.loc[mask, "quality_flag"] = QualityFlag.VALID.value
        tel.loc[mask, "sensor_status"] = SensorStatus.HEALTHY.value

        event = Event(
            event_id=event_id,
            event_type=EventType.LIGHTS_LEFT_ON,
            asset_id=asset_id,
            start_time=resolved_start.to_pydatetime() if hasattr(resolved_start, "to_pydatetime") else resolved_start,
            end_time=resolved_end.to_pydatetime() if hasattr(resolved_end, "to_pydatetime") else resolved_end,
            severity=Severity.MEDIUM,
            description=f"Classroom/office lights forced ON at rated capacity ({forced_kw:.1f} kW) during off hours ({asset_id}).",
            injection_parameters={
                "fault_type": "lights_left_on",
                "rated_power_kw": rated_power_kw,
                "forced_power_kw": forced_kw,
                "duration_hours": duration_hours,
            },
        )

    elif clean_type == "pump_no_flow":
        rated_pump_kw = float(kwargs.get("rated_power_kw", 7.5))
        # Ensure pump is electrically running
        cur_powers = tel.loc[mask, "power_kw"].fillna(rated_pump_kw)
        tel.loc[mask, "power_kw"] = cur_powers.apply(lambda p: max(float(p), rated_pump_kw))
        tel.loc[mask, "interval_energy_kwh"] = tel.loc[mask, "power_kw"] / 12.0
        tel.loc[mask, "flow_lpm"] = 0.0
        tel.loc[mask, "state"] = EquipmentState.ON.value
        tel.loc[mask, "sensor_status"] = SensorStatus.HEALTHY.value

        event = Event(
            event_id=event_id,
            event_type=EventType.PUMP_NO_FLOW,
            asset_id=asset_id,
            start_time=resolved_start.to_pydatetime() if hasattr(resolved_start, "to_pydatetime") else resolved_start,
            end_time=resolved_end.to_pydatetime() if hasattr(resolved_end, "to_pydatetime") else resolved_end,
            severity=Severity.HIGH,
            description=f"Circulation pump running dry with 0.0 LPM flow rate cavitation warning ({asset_id}).",
            injection_parameters={
                "fault_type": "pump_no_flow",
                "flow_lpm": 0.0,
                "rated_power_kw": rated_pump_kw,
                "duration_hours": duration_hours,
            },
        )

    elif clean_type == "pipe_leak_overnight":
        surge_lpm = float(kwargs.get("surge_lpm", 35.0))
        if "flow_lpm" not in tel.columns:
            tel["flow_lpm"] = 0.0
        base_flow = tel.loc[mask, "flow_lpm"].fillna(0.0)
        tel.loc[mask, "flow_lpm"] = (base_flow + surge_lpm).round(2)
        tel.loc[mask, "quality_flag"] = QualityFlag.VALID.value

        event = Event(
            event_id=event_id,
            event_type=EventType.PIPE_LEAK_OVERNIGHT,
            asset_id=asset_id,
            start_time=resolved_start.to_pydatetime() if hasattr(resolved_start, "to_pydatetime") else resolved_start,
            end_time=resolved_end.to_pydatetime() if hasattr(resolved_end, "to_pydatetime") else resolved_end,
            severity=Severity.HIGH,
            description=f"Sudden water flow surge of +{surge_lpm:.0f} LPM during off-duty hours indicating pipe rupture ({asset_id}).",
            injection_parameters={
                "fault_type": "pipe_leak_overnight",
                "surge_lpm": surge_lpm,
                "duration_hours": duration_hours,
            },
        )

    elif clean_type == "sensor_dropout":
        tel.loc[mask, "power_kw"] = None
        tel.loc[mask, "interval_energy_kwh"] = None
        tel.loc[mask, "voltage_v"] = None
        tel.loc[mask, "current_a"] = None
        tel.loc[mask, "quality_flag"] = QualityFlag.MISSING.value
        tel.loc[mask, "sensor_status"] = SensorStatus.OFFLINE.value

        event = Event(
            event_id=event_id,
            event_type=EventType.SENSOR_DROPOUT,
            asset_id=asset_id,
            start_time=resolved_start.to_pydatetime() if hasattr(resolved_start, "to_pydatetime") else resolved_start,
            end_time=resolved_end.to_pydatetime() if hasattr(resolved_end, "to_pydatetime") else resolved_end,
            severity=Severity.LOW,
            description=f"Sensor dropout: transmitter communication failure and missing quality flag ({asset_id}).",
            injection_parameters={
                "fault_type": "sensor_dropout",
                "quality_flag": "missing",
                "sensor_status": "offline",
                "duration_hours": duration_hours,
            },
        )

    elif clean_type == "pf_drop":
        drop_pf = float(kwargs.get("power_factor", 0.72))
        tel.loc[mask, "power_factor"] = round(drop_pf, 3)
        # Power factor drop increases current demand (I = P / (V * PF * sqrt(3)))
        if "power_kw" in tel.columns and "voltage_v" in tel.columns and "current_a" in tel.columns:
            valid_pwr = tel.loc[mask, "power_kw"].fillna(50.0)
            valid_volt = tel.loc[mask, "voltage_v"].fillna(415.0).replace(0, 415.0)
            tel.loc[mask, "current_a"] = ((valid_pwr * 1000.0) / (np.sqrt(3) * valid_volt * drop_pf)).round(1)

        event = Event(
            event_id=event_id,
            event_type=EventType.PF_DROP,
            asset_id=asset_id,
            start_time=resolved_start.to_pydatetime() if hasattr(resolved_start, "to_pydatetime") else resolved_start,
            end_time=resolved_end.to_pydatetime() if hasattr(resolved_end, "to_pydatetime") else resolved_end,
            severity=Severity.HIGH,
            description=f"Power factor dropped to {drop_pf:.2f}, breaching 0.90 OERC regulatory penalty threshold ({asset_id}).",
            injection_parameters={
                "fault_type": "pf_drop",
                "power_factor": drop_pf,
                "oerc_threshold": 0.90,
                "duration_hours": duration_hours,
            },
        )

    # Register into active fault registry
    active_fault_registry.register_fault(
        event=event,
        original_slice=original_slice,
        metadata={
            "fault_type": clean_type,
            "asset_id": asset_id,
            "rows_affected": int(mask.sum()),
        },
    )

    return tel, event


def clear_fault_registry() -> None:
    """Clear all active faults from the global registry."""
    active_fault_registry.clear()


def get_active_faults() -> list[Event]:
    """Retrieve all currently active live faults."""
    return active_fault_registry.get_active_faults()


def resolve_fault(event_id: str) -> bool:
    """Mark a registered fault as resolved."""
    return active_fault_registry.resolve_fault(event_id)
