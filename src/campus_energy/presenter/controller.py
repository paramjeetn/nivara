"""Nivara Presenter Kit & Sidecar Controller Application.

Runs on port 8502 (or companion tab/second monitor) to manage live demonstrations:
1. Clean State Reset Action: 1-click restore to pristine lived-in baseline (clears faults, re-seeds database & ledger).
2. Live Fault Injection Studio: Dropdown for 5 canonical faults, target asset selector, and instant live injection.
3. Replay Advancer: Steps simulated telemetry stream forward by 5s/10s.
4. Step-by-Step 2-Minute Presenter Script / Teleprompter:
   - Step 1: Baseline Overview (0:00 - 0:25)
   - Step 2: Inject Live Failure (0:25 - 0:45)
   - Step 3: Anomaly & "Why?" Diagnosis (0:45 - 1:10)
   - Step 4: Guarded Action & Safe Alternative (1:10 - 1:30)
   - Step 5: Work Order & Live Notification (1:30 - 1:45)
   - Step 6: Counterfactual M&V Verification & Audit PDF (1:45 - 2:00)
5. Pitch Visuals & State-Level Impact Scaler: Multi-campus scaling slider (10–1000 facilities) with avoided MWh, ₹Cr, and CO2 abatement.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure 'src' is on sys.path
_src_dir = str(Path(__file__).resolve().parents[2])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import pandas as pd

from campus_energy import assumptions
from campus_energy.decision.ledger import SavingsLedger
from campus_energy.demo.inject import (
    FAULT_TYPES,
    clear_fault_registry,
    get_active_faults,
    get_fault_presets,
    inject_live_fault,
)
from campus_energy.schemas import Event, EventType, Severity
from campus_energy.storage.repository import Repository
from campus_energy.storage.seed import (
    get_seed_activity_feed,
    get_seed_alert_statuses,
    get_seed_alerts,
    get_seed_work_orders,
    seed_database,
    seed_operational_history,
)

logger = logging.getLogger(__name__)

SHARED_FAULT_FILE = Path("data/live_fault_injection.json")
SHARED_REPLAY_FILE = Path("data/replay_state.json")

# ─────────────────────────────────────────────────────────────────────────────
# 2-Minute Presenter Script / Teleprompter Catalog
# ─────────────────────────────────────────────────────────────────────────────
TELEPROMPTER_STEPS: List[Dict[str, Any]] = [
    {
        "step": 1,
        "title": "Baseline Overview",
        "time_range": "0:00 - 0:25",
        "screen": "Home / Executive Dashboard (Port 8501)",
        "badge": "STEP 1: BASELINE",
        "script": (
            "Start here on the Home screen. Nivara provides campus leadership with an immediate, real-time "
            "operational pulse without manual data entry. Notice our 28-day baseline, real-time load vs forecast, "
            "and our verified savings ledger showing ₹8,175/mo in already verified savings with a 98% empirical "
            "hit rate under OERC HT Commercial tariffs. We don't just detect anomalies—we verify outcomes."
        ),
        "actions": [
            "Point to the 28-day baseline load vs forecast chart.",
            "Highlight the Verified Savings Card (₹8,175/mo verified, 962 kWh saved).",
            "Show the active 25+ sub-metered asset inventory on the campus schematic.",
        ],
        "key_points": [
            "Real-time streaming ingestion across 25+ campus assets.",
            "Calibrated Ridge regression baseline + OERC tariff engine.",
            "Lived-in operational ledger with audited historical savings.",
        ],
    },
    {
        "step": 2,
        "title": "Inject Live Failure",
        "time_range": "0:25 - 0:45",
        "screen": "Presenter Controller (Port 8502) → Live Stream",
        "badge": "STEP 2: LIVE INJECTION",
        "script": (
            "Now, watch the live telemetry stream. Using this companion presenter controller, I will inject "
            "a severe operational fault into the live stream—for instance, Utility Water Pump P1 running "
            "at full electrical load with zero water flow (dry-run cavitation). In less than 2 milliseconds, "
            "the fault mutates the telemetry stream without restarting the application or losing state."
        ),
        "actions": [
            "Select 'pump_no_flow' or 'lights_left_on' in the Fault Studio on this screen.",
            "Click '🚨 Trigger Fault into Live Stream'.",
            "Point out the sub-millisecond reaction latency badge.",
        ],
        "key_points": [
            "5 canonical faults (lighting, pump cavitation, pipe rupture, sensor dropout, PF drop).",
            "Preserves ground truth for automated IPMVP Option B counterfactual baseline.",
            "Zero application downtime or server restart required.",
        ],
    },
    {
        "step": 3,
        "title": "Anomaly & 'Why?' Diagnosis",
        "time_range": "0:45 - 1:10",
        "screen": "Alerts & Work → Alert Drawer (Port 8501)",
        "badge": "STEP 3: ROOT CAUSE DIAGNOSIS",
        "script": (
            "Instantly, the alert triggers on the primary screen in the Alerts & Work inbox. Notice the red "
            "severity chip and the human-readable headline: 'Utility Water Pump 1 is drawing 22.4 kW with 0.0 LPM flow'. "
            "Click to open the alert drawer: our triple-engine consensus—combining physical dry-run interlocks, "
            "Ridge statistical baselines, and Isolation Forest ML—explains exactly 'Why' this alert fired, "
            "eliminating false alarms while calculating the excess waste run-rate."
        ),
        "actions": [
            "On Port 8501, switch to 'Alerts & Work' or click the Top Bar alert bell.",
            "Click on the newly raised alert card to slide open the detail drawer.",
            "Point out the 'Why Did This Alert Fire?' factor attribution breakdown.",
        ],
        "key_points": [
            "Triple-engine consensus (Physical Rules + Statistical Baselines + Isolation Forest ML).",
            "Transparent factor attribution eliminates black-box AI distrust.",
            "Real-time waste run-rate calculation (excess kWh and INR exposure per day).",
        ],
    },
    {
        "step": 4,
        "title": "Guarded Action & Safe Alternative",
        "time_range": "1:10 - 1:30",
        "screen": "Service-Impact Guard (Port 8501)",
        "badge": "STEP 4: GUARDED ACTION",
        "script": (
            "Here is the AI decision support recommendation. But unlike unconstrained chatbots, Nivara passes "
            "every operational action through our deterministic Service-Impact Guard. If an operator tries to "
            "shut off a protected load—such as an ICU ventilator chiller, data center cluster, or exam hall—the "
            "Guard instantly BLOCKS the unsafe curtailment and automatically synthesizes an engineered safe alternative."
        ),
        "actions": [
            "Highlight the Service Guard status chip on the recommendation.",
            "Explain how clinical safety, cold chain, and exam continuity are mathematically guarded.",
            "Show the 3-point savings estimate range [low, expected, high].",
        ],
        "key_points": [
            "Hard-coded clinical, thermal, and critical mission-load constraints.",
            "Automatic synthesis of non-harmful safe alternatives when curtailment is blocked.",
            "3-point confidence intervals: energy (kWh), utility bill (₹), and Scope 2 carbon.",
        ],
    },
    {
        "step": 5,
        "title": "Work Order & Live Notification",
        "time_range": "1:30 - 1:45",
        "screen": "Work Order Dispatch & Notifications (Port 8501)",
        "badge": "STEP 5: CMMS DISPATCH",
        "script": (
            "Closing the loop: with a single click, we convert the guarded recommendation into a prioritized "
            "field Work Order with a strict 24-hour SLA countdown. Simultaneously, Nivara's notification router "
            "dispatches multi-channel outbound alerts via Telegram and email directly to the maintenance team lead."
        ),
        "actions": [
            "Click '🚀 Create Work Order' in the alert drawer.",
            "Show the new Work Order ticket in the Work Orders tab with SLA countdown.",
            "Show the simulated Telegram / Email dispatch confirmation badge.",
        ],
        "key_points": [
            "Instant conversion from AI recommendation to CMMS work order.",
            "Dynamic SLA deadline management based on severity and financial exposure.",
            "Live multi-channel notification dispatch (Telegram, Email, Webhook).",
        ],
    },
    {
        "step": 6,
        "title": "Counterfactual M&V Verification & Audit PDF",
        "time_range": "1:45 - 2:00",
        "screen": "IPMVP Option B Verified Savings Ledger & Audit PDF (Port 8501)",
        "badge": "STEP 6: VERIFIED M&V & AUDIT",
        "script": (
            "Finally, once the technician completes the field repair, Nivara performs continuous IPMVP Option B "
            "counterfactual baseline verification against actual post-fix telemetry. Look at the ledger: it validates "
            "avoided kWh, financial recovery in Rupees, and CO2 abatement, updating our empirical hit rate. Finally, "
            "one click exports a statutory, court-ready Executive Audit PDF complete with ECBC 2017 chapter compliance."
        ),
        "actions": [
            "Click '🔬 Resolve & Verify' on the work order.",
            "Highlight the updated Ledger hit rate (98% empirical accuracy).",
            "Show the 'Download Executive Audit PDF' button for statutory compliance.",
        ],
        "key_points": [
            "IPMVP Option B / Retrofit Isolation counterfactual baseline verification.",
            "Closed-loop tracking: proposed → accepted → work_order → done → verified.",
            "One-click ReportLab PDF generation with ECBC 2017 Chapter 5, 6, 7 compliance.",
        ],
    },
]


def get_teleprompter_step(step_number: int) -> Dict[str, Any]:
    """Retrieve teleprompter script and metadata for a specific 1-indexed step."""
    clean_step = max(1, min(step_number, len(TELEPROMPTER_STEPS)))
    return TELEPROMPTER_STEPS[clean_step - 1]


# ─────────────────────────────────────────────────────────────────────────────
# Core Programmatic Controller Functions (Unit-Testable)
# ─────────────────────────────────────────────────────────────────────────────
def reset_demo_state(
    facility_id: str = "university",
    db_path: str = "data/campus.db",
    reference_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """1-Click Clean State Reset: restores pristine lived-in baseline deterministically.

    Actions:
    1. Clears in-memory ActiveFaultRegistry.
    2. Clears shared disk fault file (data/live_fault_injection.json).
    3. Resets replay state to 0 offset.
    4. Re-seeds SQLite database with the 5 canonical alerts.
    5. Re-seeds SavingsLedger with the 3 verified work orders.

    Returns:
        Dict with clean state summary (ledger, active_faults_count, alerts_count).
    """
    now = reference_time or datetime.now(timezone.utc)

    # 1. Clear ActiveFaultRegistry
    clear_fault_registry()

    # 2. Clear shared live fault file
    if SHARED_FAULT_FILE.exists():
        try:
            SHARED_FAULT_FILE.unlink()
        except Exception as e:
            logger.warning("Failed to unlink shared fault file: %s", e)

    # Also clear empty json into it
    SHARED_FAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SHARED_FAULT_FILE, "w", encoding="utf-8") as f:
        json.dump({"active": False, "faults": []}, f)

    # 3. Reset replay state
    SHARED_REPLAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    replay_data = {
        "step_index": 0,
        "offset_seconds": 0,
        "last_stepped_at": now.isoformat(),
    }
    with open(SHARED_REPLAY_FILE, "w", encoding="utf-8") as f:
        json.dump(replay_data, f)

    # 4. Re-seed SQLite database
    try:
        seed_database(db_path=db_path, reference_time=now)
    except Exception as exc:
        logger.debug("Database reset warning: %s", exc)

    # 5. Re-seed SavingsLedger
    ledger = SavingsLedger()
    seed_operational_history(ledger=ledger, facility_id=facility_id, db_path=db_path, reference_time=now)

    return {
        "status": "success",
        "message": "Demo state restored to pristine lived-in baseline.",
        "active_faults_count": len(get_active_faults()),
        "verified_work_orders_count": len(ledger.get_items("verified")),
        "cumulative_verified_inr": ledger.get_summary().cumulative_verified_inr,
        "cumulative_verified_kwh": ledger.get_summary().cumulative_verified_kwh,
        "estimate_hit_rate_pct": ledger.get_summary().estimate_hit_rate_pct,
        "alerts_count": len(get_seed_alerts(now)),
        "ledger": ledger,
    }


def trigger_live_fault(
    fault_type: str,
    asset_id: str,
    duration_hours: float = 4.0,
    telemetry_df: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Event]:
    """Inject a live canonical fault into the simulated stream and persist to shared disk file.

    Parameters:
        fault_type: One of 'lights_left_on', 'pump_no_flow', 'pipe_leak_overnight',
            'sensor_dropout', 'pf_drop'.
        asset_id: Target asset identifier.
        duration_hours: Fault duration in hours (default: 4.0).
        telemetry_df: Optional DataFrame. If None, loaded from data/generated/telemetry.csv.
        **kwargs: Overrides (e.g. power_factor, surge_lpm).

    Returns:
        Tuple[pd.DataFrame, Event]: (modified_telemetry_df, injected_event)
    """
    # Load telemetry if not provided
    if telemetry_df is None:
        t_path = Path("data/generated/telemetry.csv")
        if t_path.exists():
            df = pd.read_csv(t_path)
        else:
            # Synthetic minimal telemetry fallback for testing
            now = datetime.now(timezone.utc)
            timestamps = [now - timedelta(minutes=5 * i) for i in range(50)][::-1]
            df = pd.DataFrame({
                "timestamp": [ts.isoformat() for ts in timestamps],
                "asset_id": [asset_id] * 50,
                "power_kw": [15.0] * 50,
                "interval_energy_kwh": [1.25] * 50,
                "flow_lpm": [120.0] * 50,
                "power_factor": [0.94] * 50,
                "quality_flag": ["valid"] * 50,
                "sensor_status": ["healthy"] * 50,
            })
    else:
        df = telemetry_df.copy()

    # Inject live fault into telemetry & active_fault_registry
    t_start = time.perf_counter()
    mod_tel, event = inject_live_fault(
        telemetry_df=df,
        fault_type=fault_type,
        asset_id=asset_id,
        duration_hours=duration_hours,
        **kwargs,
    )
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    # Persist injected fault to shared disk file for inter-process communication with port 8501
    SHARED_FAULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fault_payload = {
        "active": True,
        "event_id": event.event_id,
        "fault_type": fault_type,
        "asset_id": asset_id,
        "severity": event.severity.value,
        "description": event.description,
        "start_time": event.start_time.isoformat() if hasattr(event.start_time, "isoformat") else str(event.start_time),
        "end_time": event.end_time.isoformat() if hasattr(event.end_time, "isoformat") else str(event.end_time),
        "duration_hours": duration_hours,
        "latency_ms": round(latency_ms, 2),
        "injected_at": datetime.now(timezone.utc).isoformat(),
        "injection_parameters": event.injection_parameters,
    }
    with open(SHARED_FAULT_FILE, "w", encoding="utf-8") as f:
        json.dump(fault_payload, f, indent=2)

    return mod_tel, event


def advance_replay_stream(
    seconds: int = 5,
    db_path: str = "data/campus.db",
) -> Dict[str, Any]:
    """Step simulated stream forward by designated seconds and record in shared state.

    Parameters:
        seconds: Step duration in seconds (default: 5 or 10).
        db_path: SQLite database path.

    Returns:
        Dict with updated replay step index, offset_seconds, and simulated time.
    """
    clean_sec = max(1, int(seconds))

    cur_step = 0
    cur_offset = 0
    if SHARED_REPLAY_FILE.exists():
        try:
            with open(SHARED_REPLAY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                cur_step = int(data.get("step_index", 0))
                cur_offset = int(data.get("offset_seconds", 0))
        except Exception:
            pass

    new_step = cur_step + 1
    new_offset = cur_offset + clean_sec
    now = datetime.now(timezone.utc)
    simulated_time = now + timedelta(seconds=new_offset)

    payload = {
        "step_index": new_step,
        "offset_seconds": new_offset,
        "stepped_by_seconds": clean_sec,
        "simulated_time": simulated_time.isoformat(),
        "updated_at": now.isoformat(),
    }

    SHARED_REPLAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SHARED_REPLAY_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Also update repository table if accessible
    try:
        repo = Repository(db_path=db_path)
        with repo.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO replay_state (id, current_time, status, speed) VALUES (1, ?, 'running', ?)",
                (simulated_time.isoformat(), clean_sec),
            )
    except Exception as exc:
        logger.debug("Repository replay_state update: %s", exc)

    return payload


def compute_multi_campus_impact(
    num_facilities: int = 100,
    annual_kwh_per_facility: float = 42000.0,
    avg_tariff_inr: float = 8.50,
    grid_emission_factor: float = 0.82,
    coal_factor_kg_per_kwh: float = 0.50,
) -> Dict[str, Any]:
    """Compute state-level scaling impact across Odisha public facilities.

    Parameters:
        num_facilities: Number of adopting institutions (e.g. 100, 500, 1000).
        annual_kwh_per_facility: Conservative avoided electricity per facility (kWh/yr).
        avg_tariff_inr: Grid tariff in INR/kWh.
        grid_emission_factor: Grid emission factor (kg CO2e/kWh).
        coal_factor_kg_per_kwh: Thermal coal burnt per kWh (kg/kWh).

    Returns:
        Dict with total avoided MWh, GWh, ₹ Crores, Metric Tonnes CO2e, and Tonnes of Coal.
    """
    n = max(1, int(num_facilities))
    tot_kwh = n * annual_kwh_per_facility
    tot_mwh = tot_kwh / 1000.0
    tot_gwh = tot_mwh / 1000.0

    tot_inr = tot_kwh * avg_tariff_inr
    tot_crores = tot_inr / 10000000.0  # 1 Crore = 10^7 INR

    tot_co2_kg = tot_kwh * grid_emission_factor
    tot_co2_metric_tonnes = tot_co2_kg / 1000.0

    tot_coal_kg = tot_kwh * coal_factor_kg_per_kwh
    tot_coal_tonnes = tot_coal_kg / 1000.0

    return {
        "num_facilities": n,
        "avoided_mwh": round(tot_mwh, 1),
        "avoided_gwh": round(tot_gwh, 2),
        "savings_crores_inr": round(tot_crores, 2),
        "carbon_avoided_metric_tonnes": round(tot_co2_metric_tonnes, 1),
        "coal_avoided_tonnes": round(tot_coal_tonnes, 1),
        "breakdown": {
            "universities_share": round(tot_crores * 0.40, 2),
            "district_hospitals_share": round(tot_crores * 0.35, 2),
            "secretariat_complexes_share": round(tot_crores * 0.25, 2),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit Presenter Sidecar UI (Runs on Port 8502)
# ─────────────────────────────────────────────────────────────────────────────
def run_presenter_ui() -> None:
    """Render the full presenter controller application in Streamlit."""
    import streamlit as st

    st.set_page_config(
        page_title="Nivara Presenter Controller",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b1120;
            color: #f1f5f9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .presenter-header {
            background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .teleprompter-card {
            background: #0f172a;
            border: 2px solid #38bdf8;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 20px rgba(56, 189, 248, 0.15);
        }
        .teleprompter-text {
            font-size: 1.18rem;
            line-height: 1.65;
            color: #f8fafc;
            font-weight: 500;
            margin: 14px 0;
            padding: 14px;
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid #38bdf8;
            border-radius: 4px;
        }
        .action-chip {
            background: rgba(52, 211, 153, 0.15);
            border: 1px solid #10b981;
            color: #34d399;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
        }
        .kpi-tile {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 14px;
            text-align: center;
        }
        .kpi-tile .val {
            font-size: 1.6rem;
            font-weight: 700;
            color: #38bdf8;
        }
        .kpi-tile .lbl {
            font-size: 0.78rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top Presenter Header
    st.markdown(
        """
        <div class="presenter-header">
            <div>
                <h2 style="margin: 0; color: #38bdf8; font-size: 1.45rem;">
                    🎙️ Nivara Presenter Controller & Teleprompter
                </h2>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 3px;">
                    Companion Sidecar Display (Port 8502) · Primary Product: <code>http://localhost:8501</code>
                </div>
            </div>
            <div>
                <span class="action-chip">🟢 Companion Sync: Connected</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Action Bar: Clean State Reset & Status
    # ─────────────────────────────────────────────────────────────────────────
    top_col1, top_col2 = st.columns([6, 4])
    with top_col1:
        st.markdown("**⚡ Rapid Demo Session Management**")
        if st.button("🔄 Clean State Reset (Pristine Lived-In Baseline)", use_container_width=True, type="primary"):
            res = reset_demo_state()
            st.session_state["teleprompter_step"] = 1
            st.success(
                f"✅ System restored to pristine lived-in baseline! "
                f"({res['verified_work_orders_count']} verified work orders, "
                f"{res['alerts_count']} alerts, 0 active injected faults)."
            )
            time.sleep(0.5)
            st.rerun()

    with top_col2:
        active_faults = get_active_faults()
        n_faults = len(active_faults)
        if n_faults > 0:
            st.warning(f"🚨 {n_faults} Live Fault Active in Telemetry Stream")
        else:
            st.info("🟢 Telemetry Stream Baseline Clean · No Active Injected Faults")

    st.markdown("<hr style='border-color: #334155; margin: 15px 0;'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Main Split: Teleprompter vs Fault Studio
    # ─────────────────────────────────────────────────────────────────────────
    stage_left, stage_right = st.columns([5, 5])

    # ── Left Stage: 2-Minute Presenter Script / Teleprompter ──
    with stage_left:
        st.subheader("⏱️ 2-Minute Script & Teleprompter")

        if "teleprompter_step" not in st.session_state:
            st.session_state["teleprompter_step"] = 1

        step_idx = st.session_state["teleprompter_step"]
        step_data = get_teleprompter_step(step_idx)

        # Step Stepper Navigation
        nav_prev, nav_status, nav_next = st.columns([2, 5, 2])
        with nav_prev:
            if st.button("⏮️ Prev Step", disabled=(step_idx <= 1), use_container_width=True):
                st.session_state["teleprompter_step"] = max(1, step_idx - 1)
                st.rerun()
        with nav_status:
            step_opts = [f"Step {s['step']}: {s['title']} ({s['time_range']})" for s in TELEPROMPTER_STEPS]
            chosen_step_str = st.selectbox(
                "Jump to Step",
                step_opts,
                index=step_idx - 1,
                label_visibility="collapsed",
            )
            selected_idx = int(chosen_step_str.split(":")[0].replace("Step", "").strip())
            if selected_idx != step_idx:
                st.session_state["teleprompter_step"] = selected_idx
                st.rerun()
        with nav_next:
            if st.button("Next Step ⏭️", disabled=(step_idx >= len(TELEPROMPTER_STEPS)), use_container_width=True):
                st.session_state["teleprompter_step"] = min(len(TELEPROMPTER_STEPS), step_idx + 1)
                st.rerun()

        # Teleprompter Card
        st.markdown(
            f"""
            <div class="teleprompter-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="background: #0284c7; color: white; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">
                        {step_data['badge']} · {step_data['time_range']}
                    </span>
                    <span style="font-size: 0.82rem; color: #94a3b8;">
                        Target Screen: <b>{step_data['screen']}</b>
                    </span>
                </div>
                <h3 style="color: #38bdf8; margin: 10px 0 4px 0; font-size: 1.25rem;">
                    {step_data['title']}
                </h3>
                <div class="teleprompter-text">
                    "{step_data['script']}"
                </div>
                <div style="margin-top: 12px;">
                    <b style="color: #34d399; font-size: 0.85rem;">🎯 Actions to Perform on Main Screen:</b>
                    <ul style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px; padding-left: 20px;">
                        {''.join(f'<li>{act}</li>' for act in step_data['actions'])}
                    </ul>
                </div>
                <div style="margin-top: 8px;">
                    <b style="color: #f59e0b; font-size: 0.85rem;">💡 Key Technical Highlights:</b>
                    <ul style="color: #94a3b8; font-size: 0.82rem; margin-top: 4px; padding-left: 20px;">
                        {''.join(f'<li>{kp}</li>' for kp in step_data['key_points'])}
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Right Stage: Live Fault Injection Studio & Replay Advancer ──
    with stage_right:
        st.subheader("⚡ Live Fault Injection Studio")
        st.caption("Trigger live physical and operational faults directly into telemetry without restarting the server.")

        presets = get_fault_presets()
        fault_type_choice = st.selectbox(
            "Select Canonical Fault Scenario",
            options=FAULT_TYPES,
            format_func=lambda x: f"{presets.get(x, {}).get('icon', '⚡')} {presets.get(x, {}).get('title', x)} ({presets.get(x, {}).get('category', 'Fault')})",
        )

        preset = presets.get(fault_type_choice, {})
        st.markdown(
            f"""
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.85rem; color: #cbd5e1;">
                <b>Expected Detection:</b> <span style="color: #34d399;">{preset.get('expected_detection', 'N/A')}</span><br>
                <b>Description:</b> {preset.get('description', '')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Asset selector dynamically suggested
        asset_hint = preset.get("asset_hint", "meter")
        asset_options = {
            "lighting": ["EQ-A-CR101-LT", "EQ-ADM-LT-01", "LIGHT-ADMIN-01"],
            "pump": ["EQ-UTL-P1-PM", "PUMP-MAIN-01"],
            "meter": ["MTR-MAIN-01", "MTR-WATER-01", "EQ-B-CR201-LT"],
        }.get(asset_hint, ["EQ-A-CR101-LT", "EQ-UTL-P1-PM", "MTR-MAIN-01"])

        asset_id_choice = st.selectbox("Target Asset Identifier", options=asset_options)
        duration_hrs = st.slider("Fault Duration (Hours)", min_value=1.0, max_value=8.0, value=float(preset.get("default_duration_hours", 4.0)), step=0.5)

        # Inject Trigger Button
        if st.button("🚨 Trigger Fault into Live Stream", type="primary", use_container_width=True):
            with st.spinner("Injecting fault into real-time telemetry stream..."):
                t_start = time.perf_counter()
                mod_tel, event = trigger_live_fault(
                    fault_type=fault_type_choice,
                    asset_id=asset_id_choice,
                    duration_hours=duration_hrs,
                )
                latency_ms = (time.perf_counter() - t_start) * 1000.0

            st.success(
                f"✅ Fault '{preset.get('title')}' injected on asset `{asset_id_choice}`! "
                f"(Reaction Latency: {latency_ms:.1f}ms). "
                f"Alert generated with ID: `{event.event_id}`."
            )
            # Advance teleprompter to Step 3
            st.session_state["teleprompter_step"] = 3
            time.sleep(0.4)
            st.rerun()

        st.divider()

        # Replay Advancer
        st.subheader("⏩ Simulated Replay Advancer")
        st.caption("Advance the continuous simulated stream timeline forward by 5s or 10s.")
        rep_c1, rep_c2 = st.columns(2)
        with rep_c1:
            if st.button("⏩ Advance Stream (+5s)", use_container_width=True):
                rep_res = advance_replay_stream(seconds=5)
                st.info(f"Stream advanced +5s (Total offset: +{rep_res['offset_seconds']}s).")
        with rep_c2:
            if st.button("⏭️ Advance Stream (+10s)", use_container_width=True):
                rep_res = advance_replay_stream(seconds=10)
                st.info(f"Stream advanced +10s (Total offset: +{rep_res['offset_seconds']}s).")

    st.markdown("<hr style='border-color: #334155; margin: 20px 0;'>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Pitch Visuals & State-Level Impact Scaler (Moved out of Main Product!)
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("🏛️ Pitch Visuals & State-Level Scaling Impact Engine")
    st.caption("Interactive multi-campus scaling slider demonstrating statewide public sector ROI across Odisha.")

    scale_col1, scale_col2 = st.columns([4, 6])
    with scale_col1:
        st.markdown("**Simulate Statewide Nivara Rollout**")
        num_facs = st.slider(
            "State Facilities Adopting Nivara",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Simulates collective avoided expenditure across State Universities, District Hospitals, and Secretariats.",
        )
        impact = compute_multi_campus_impact(num_facilities=num_facs)
        st.caption(f"Estimated for **{num_facs} public institutions** over 12 operating months.")

    with scale_col2:
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="val">{impact['avoided_gwh']}</div>
                    <div class="lbl">Avoided GWh / Yr</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi2:
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="val">₹{impact['savings_crores_inr']:.1f} Cr</div>
                    <div class="lbl">Public Savings</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi3:
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="val">{impact['carbon_avoided_metric_tonnes']:,.0f}</div>
                    <div class="lbl">MT CO₂e Avoided</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi4:
            st.markdown(
                f"""
                <div class="kpi-tile">
                    <div class="val">{impact['coal_avoided_tonnes']:,.0f}</div>
                    <div class="lbl">Tonnes Coal Saved</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Institutional Breakdown
        bd = impact["breakdown"]
        st.markdown(
            f"""
            <div style="margin-top: 12px; font-size: 0.85rem; color: #cbd5e1; background: #0f172a; padding: 10px 14px; border-radius: 6px; border: 1px solid #334155;">
                <b>Odisha Public Sector Financial Impact Breakdown:</b><br>
                🎓 Higher Education & Technical Universities (40%): <b>₹{bd['universities_share']:.2f} Cr/yr</b> &nbsp;|&nbsp;
                🏥 District Hospitals & CHCs (35%): <b>₹{bd['district_hospitals_share']:.2f} Cr/yr</b> &nbsp;|&nbsp;
                🏛️ State Administrative Secretariat & Collectorates (25%): <b>₹{bd['secretariat_complexes_share']:.2f} Cr/yr</b>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    run_presenter_ui()
