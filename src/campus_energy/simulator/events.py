"""Event injection engine for the campus simulator.

Applies operational anomalies and sensor faults to the normal baseline data.
Now facility-aware: resolves asset IDs dynamically from the asset registry
rather than using university-hardcoded IDs.
"""
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from campus_energy.schemas import EventType, Severity


def _pick_asset(assets_df: pd.DataFrame, asset_type: str, exclude: list[str] | None = None) -> str | None:
    """Pick the first available asset of a given type, excluding already-used ones."""
    mask = assets_df["asset_type"] == asset_type
    if exclude:
        mask &= ~assets_df["asset_id"].isin(exclude)
    subset = assets_df[mask]
    if subset.empty:
        return None
    return subset.iloc[0]["asset_id"]


def _pick_pump(assets_df: pd.DataFrame, used: list[str]) -> str | None:
    return _pick_asset(assets_df, "pump", used)


def inject_events(
    telemetry_df: pd.DataFrame,
    context_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Inject anomalies into the normal telemetry and context data.

    Returns (modified_telemetry_df, events_df, ground_truth_df).
    """
    rng = np.random.default_rng(seed)

    tel = telemetry_df.copy()
    tel["timestamp"] = pd.to_datetime(tel["timestamp"])
    ctx = context_df.copy()
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"])

    events = []
    ground_truth = []
    used_assets: list[str] = []

    def add_event(event_type, asset_id, start, end, severity, desc, gt_method, root_cause, params=None):
        event_id = f"EVT-{uuid.uuid4().hex[:8].upper()}"
        events.append({
            "event_id": event_id,
            "event_type": event_type,
            "asset_id": asset_id,
            "start_time": start.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            "end_time": end.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            "severity": severity,
            "description": desc,
            "injection_parameters": params or {},
        })
        ground_truth.append({
            "event_id": event_id,
            "asset_id": asset_id,
            "event_type": event_type,
            "start_time": start.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            "end_time": end.strftime("%Y-%m-%dT%H:%M:%S+05:30"),
            "severity": severity,
            "expected_detection": True,
            "expected_detection_method": gt_method,
            "root_cause": root_cause,
        })
        return event_id

    def get_mask(asset_id, start, end):
        return (
            (tel["asset_id"] == asset_id)
            & (tel["timestamp"] >= start)
            & (tel["timestamp"] <= end)
        )

    sim_start = tel["timestamp"].min()

    # ── Equipment-level assets by type ──────────────────────────────────────
    equip = assets_df[assets_df["monitoring_level"] == "equipment"]

    # ---------------------------------------------------------
    # A. Lights left ON after hours (any lighting asset)
    # ---------------------------------------------------------
    target_lights = _pick_asset(equip, "lighting", used_assets)
    if target_lights is not None:
        used_assets.append(target_lights)
        start_A = sim_start + timedelta(days=2, hours=18)
        end_A = start_A + timedelta(hours=6)
        rated_A_rows = assets_df[assets_df["asset_id"] == target_lights]["rated_power_kw"]
        if not rated_A_rows.empty and pd.notna(rated_A_rows.iloc[0]):
            rated_A = float(rated_A_rows.iloc[0])
            mask_A = get_mask(target_lights, start_A, end_A)
            tel.loc[mask_A, "power_kw"] = rated_A * 0.95
            tel.loc[mask_A, "interval_energy_kwh"] = (rated_A * 0.95) / 12.0
            tel.loc[mask_A, "state"] = "ON"
            add_event(
                EventType.LIGHTS_LEFT_ON, target_lights, start_A, end_A, Severity.MEDIUM,
                f"Lights left on overnight ({target_lights})", "rule", "User forgot to switch off"
            )

    # ---------------------------------------------------------
    # B. HVAC left ON after hours (any hvac asset)
    # ---------------------------------------------------------
    target_hvac = _pick_asset(equip, "hvac", used_assets)
    if target_hvac is not None:
        used_assets.append(target_hvac)
        start_B = sim_start + timedelta(days=5, hours=19)
        end_B = start_B + timedelta(hours=8)
        rated_B_rows = assets_df[assets_df["asset_id"] == target_hvac]["rated_power_kw"]
        if not rated_B_rows.empty and pd.notna(rated_B_rows.iloc[0]):
            rated_B = float(rated_B_rows.iloc[0])
            mask_B = get_mask(target_hvac, start_B, end_B)
            tel.loc[mask_B, "power_kw"] = rated_B * 0.4
            tel.loc[mask_B, "interval_energy_kwh"] = (rated_B * 0.4) / 12.0
            tel.loc[mask_B, "state"] = "ON"
            add_event(
                EventType.HVAC_LEFT_ON, target_hvac, start_B, end_B, Severity.MEDIUM,
                f"HVAC running after hours ({target_hvac})", "rule", "Timer override active"
            )

    # ---------------------------------------------------------
    # C. Pump running without water flow (any pump)
    # ---------------------------------------------------------
    target_pump = _pick_pump(equip, used_assets)
    if target_pump is not None:
        used_assets.append(target_pump)
        start_C = sim_start + timedelta(days=8, hours=11, minutes=30)
        end_C = start_C + timedelta(hours=1, minutes=30)
        mask_C = get_mask(target_pump, start_C, end_C)
        tel.loc[mask_C, "flow_lpm"] = 0.0
        add_event(
            EventType.PUMP_NO_FLOW, target_pump, start_C, end_C, Severity.HIGH,
            f"Pump running dry ({target_pump})", "rule", "Valve closed downstream"
        )

    # ---------------------------------------------------------
    # I. Sensor dropout (second lighting asset)
    # ---------------------------------------------------------
    target_drop = _pick_asset(equip, "lighting", used_assets)
    if target_drop is not None:
        used_assets.append(target_drop)
        start_I = sim_start + timedelta(days=12, hours=10)
        end_I = start_I + timedelta(hours=3)
        mask_I = get_mask(target_drop, start_I, end_I)
        tel.loc[mask_I, "power_kw"] = None
        tel.loc[mask_I, "interval_energy_kwh"] = None
        tel.loc[mask_I, "quality_flag"] = "missing"
        tel.loc[mask_I, "sensor_status"] = "offline"
        add_event(
            EventType.SENSOR_DROPOUT, target_drop, start_I, end_I, Severity.LOW,
            f"Lighting meter stopped reporting ({target_drop})", "rule", "Network interference"
        )

    # ---------------------------------------------------------
    # M. Duplicate reading (any computer_lab asset)
    # ---------------------------------------------------------
    target_dup = _pick_asset(equip, "computer_lab", used_assets)
    if target_dup is not None:
        used_assets.append(target_dup)
        start_M = sim_start + timedelta(days=15, hours=14)
        mask_M = get_mask(target_dup, start_M, start_M)
        if mask_M.any():
            dup_row = tel[mask_M].copy()
            dup_row["reading_id"] = dup_row["reading_id"] + "-DUP"
            dup_row["quality_flag"] = "duplicated"
            tel = pd.concat([tel, dup_row], ignore_index=True)
            add_event(
                EventType.DUPLICATE_READING, target_dup, start_M, start_M + timedelta(minutes=5),
                Severity.LOW, f"Duplicate MQTT message ({target_dup})", "rule", "Network retry"
            )

    # Re-sort telemetry if we appended rows
    tel.sort_values(["timestamp", "asset_id"], inplace=True)
    tel["timestamp"] = tel["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+05:30")

    events_df = pd.DataFrame(events)
    gt_df = pd.DataFrame(ground_truth)

    return tel, events_df, gt_df
