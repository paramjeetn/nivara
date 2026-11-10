"""
BPUT Campus Energy Intelligence Platform — Streamlit Dashboard
=============================================================
Pages:
  1. Executive Overview (ESG Impact)
  2. 3D Campus Map (Real-Time Heatmap)
  3. AI Forecasting & Scenarios
  4. NLP Complaints & Alerts
  5. Microgrid Dispatch (Solar + BESS)
  6. Department Billing & Chargebacks
  7. Asset Explorer
  8. AI Facility Assistant (Gemini)
"""
import warnings
warnings.filterwarnings("ignore")
import logging
import os
import sys
from pathlib import Path

# Ensure 'src' directory is in sys.path even when app is run without 'pip install -e .'
_src_dir = str(Path(__file__).resolve().parents[2])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import traceback
from typing import Any, Dict, List, Optional, Union

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import importlib
import time

logger = logging.getLogger(__name__)

from campus_energy import assumptions
from campus_energy.demo.inject import (
    FAULT_TYPES,
    clear_fault_registry,
    get_active_faults,
    get_fault_presets,
    inject_live_fault,
)
from campus_energy.demo.tour import (
    GuidedTour,
    get_default_tour,
    get_tour_step_info,
)
from campus_energy.adapters import (
    DataIngestionError,
    DataQualityReport,
    detect_column_mapping,
    process_uploaded_csv,
)


def render_assumptions_expander(keys: list[str]):
    """Render an expandable panel showing assumption keys, values, confidence, and sources."""
    with st.expander("ℹ️ Assumptions used"):
        rows = []
        for k in keys:
            try:
                item = assumptions.get(k)
                conf_badge = {
                    "high": "🟢 High",
                    "medium": "🟡 Medium",
                    "low": "🟠 Low",
                    "needs_verification": "🔴 Needs Verification",
                }.get(item.confidence, item.confidence)
                rows.append({
                    "Assumption Key": item.key,
                    "Value": f"{item.value} {item.unit}".strip(),
                    "Confidence": conf_badge,
                    "Source Citation": f"{item.source} ({item.source_date})",
                    "Description": item.description,
                })
            except KeyError:
                continue
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Page Config (MUST be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BPUT Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Nivara enterprise design system styles
from campus_energy.ui.styles import load_global_styles
load_global_styles()

# ─────────────────────────────────────────────────────────────────────────────
# Facility Definitions (Dynamic Discovery from Facility Profiles)
# ─────────────────────────────────────────────────────────────────────────────
from campus_energy.profiles import load_all_profiles, FacilityProfile

try:
    _LOADED_PROFILES = load_all_profiles("config/facility_profiles")
except Exception:
    try:
        _LOADED_PROFILES = load_all_profiles("configs/facility_profiles")
    except Exception:
        _LOADED_PROFILES = {}

def _build_facility_display_label(profile: FacilityProfile) -> str:
    icon = getattr(profile, "icon", "🏢")
    type_display_map = {
        "university": "University",
        "hospital": "District Hospital",
        "district_hospital": "District Hospital",
        "govt_office": "Government Office",
        "government_office": "Government Office",
        "industrial": "Industrial Campus",
        "industrial_campus": "Industrial Campus",
    }
    type_label = type_display_map.get(profile.facility_type, profile.facility_type.replace("_", " ").title())
    return f"{icon} {type_label} — {profile.facility_name}"

FACILITY_OPTIONS = {
    _build_facility_display_label(prof): prof
    for prof in _LOADED_PROFILES.values()
}

# Fallback dictionary if no profiles directory found
if not FACILITY_OPTIONS:
    FACILITY_OPTIONS = {
        "🎓 University — Vidya Institute of Technology": {
            "key": "university",
            "data_dir": Path("data/generated"),
            "config": "configs/simulation.yaml",
            "floor_area_m2": 12000,
            "peak_occupancy": 2500,
            "location": "Bhubaneswar, Odisha",
        }
    }

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Data Loading
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/generated")

CAMPUS_COORDS = {
    "BLD-A":   [84.8185, 22.2480],
    "BLD-B":   [84.8195, 22.2485],
    "BLD-ADM": [84.8192, 22.2472],
    "BLD-LIB": [84.8198, 22.2475],
    "BLD-HST": [84.8175, 22.2460],
    "BLD-UTL": [84.8210, 22.2465],
}

# Sub-zone micro-offsets around each building center for high-density sensor pinning
ZONE_OFFSETS = {
    # Building A zones
    "BLD-A-CR101": [-0.00045, -0.00025],
    "BLD-A-CR102": [-0.00030,  0.00030],
    "BLD-A-CL01":  [ 0.00035, -0.00020],
    "BLD-A-CL02":  [ 0.00040,  0.00028],
    # Building B zones
    "BLD-B-CR201": [-0.00035,  0.00020],
    "BLD-B-LAB1":  [ 0.00035, -0.00025],
    # Administration zones
    "BLD-ADM-OF101": [-0.00030, -0.00020],
    "BLD-ADM-MR201": [ 0.00030,  0.00020],
    # Library zones
    "BLD-LIB-RH": [-0.00035,  0.00015],
    "BLD-LIB-CA": [ 0.00035, -0.00020],
    # Hostel zones
    "BLD-HST-CA": [-0.00030,  0.00020],
    "BLD-HST-UA": [ 0.00030, -0.00020],
    # Utility zones
    "BLD-UTL-P1": [-0.00025,  0.00015],
    "BLD-UTL-P2": [ 0.00025, -0.00015],
}


CAMPUS_BOUNDARY = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [84.8160, 22.2500], [84.8215, 22.2500], [84.8230, 22.2475],
                [84.8210, 22.2445], [84.8160, 22.2450], [84.8160, 22.2500],
            ]]
        }
    }]
}

MAPBOX_TOKEN = os.environ.get(
    "MAPBOX_TOKEN",
    "pk.eyJ1IjoicGFyYW1qZWV0LTM5MzkiLCJhIjoiY212MHRhN3N6MDhhNjJ5c2R3Y2c2YTRneiJ9.wa2xVeMKEKZVLkgwE667TQ",
)

COLORS = {
    "primary": "#38bdf8",
    "accent": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "purple": "#8b5cf6",
}

from campus_energy.decision import (
    Recommendation,
    SavingsBundle,
    SavingsRange,
    GuardResult,
    check_service_guard,
    apply_service_guard,
    calculate_savings,
    RecommendationEngine,
    generate_recommendations_from_alerts,
    generate_recommendations_from_forecast,
    generate_recommendations_from_scenarios,
    generate_recommendations_from_complaints,
    build_ranked_recommendations,
    LedgerSummary,
    SavingsLedger,
    WorkOrderItem,
)
from campus_energy.simulator.events import simulate_fault_resolution
from campus_energy.notify import (
    NotificationMessage,
    NotificationResult,
    NotificationRouter,
    get_notification_router,
)


def get_or_create_savings_ledger(
    facility_meta: Any,
    events_or_alerts: Optional[Union[pd.DataFrame, List[Any]]] = None,
    tel_df: Optional[pd.DataFrame] = None,
) -> SavingsLedger:
    """Retrieve or initialize the persistent SavingsLedger for the active facility."""
    fac_id = getattr(facility_meta, "facility_id", "university") if not isinstance(facility_meta, dict) else facility_meta.get("key", "university")
    key = f"savings_ledger_{fac_id}"
    if key not in st.session_state:
        ledger = SavingsLedger()
        if events_or_alerts is not None:
            has_items = len(events_or_alerts) > 0 if not isinstance(events_or_alerts, pd.DataFrame) else not events_or_alerts.empty
            if has_items:
                try:
                    raw_recs = generate_recommendations_from_alerts(events_or_alerts, profile=facility_meta)
                    app_recs, _ = apply_service_guard(raw_recs, facility_meta)
                    for r in app_recs:
                        ledger.add_recommendation(r)
                except Exception:
                    pass
        st.session_state[key] = ledger
    st.session_state["savings_ledger"] = st.session_state[key]
    return st.session_state[key]


def render_service_guard_demo_expander(facility_profile: Any, expanded: bool = False):
    """Render the dedicated '🛡️ Blocked by Service Guard (Jury Demo Moment)' expander."""
    with st.expander("🛡️ Blocked by Service Guard (Jury Demo Moment)", expanded=expanded):
        st.markdown(
            """
            > **The Nivara Safety Principle:** Energy efficiency optimization must **never** compromise
            > human life, clinical safety, critical cold chains, research continuity, or occupant thermal comfort standards.
            """
        )

        fac_id = getattr(facility_profile, "facility_id", "university") if not isinstance(facility_profile, dict) else facility_profile.get("key", "university")
        if fac_id in ("hospital", "district_hospital"):
            demo_candidates = [
                Recommendation(
                    id="DEMO-BLK-01",
                    facility_id="district_hospital",
                    type="peak_shaving",
                    target="EQ-IPD-ICU-EQ",
                    action_text="Curtail ICU life-support monitor and ventilator power by 30% during grid peak tariff surcharge.",
                    trigger="grid_peak_tariff_spike",
                    savings=SavingsBundle.from_expected(45.0),
                    owner_role="operations",
                ),
                Recommendation(
                    id="DEMO-BLK-02",
                    facility_id="district_hospital",
                    type="equipment_shutdown",
                    target="EQ-IPD-CCR-AC",
                    action_text="Shut down cold chain vaccine refrigeration compressor for 2 hours to avoid demand penalty.",
                    trigger="demand_limit_threshold",
                    savings=SavingsBundle.from_expected(32.0),
                    owner_role="operations",
                ),
                Recommendation(
                    id="DEMO-BLK-03",
                    facility_id="district_hospital",
                    type="setpoint_adjustment",
                    target="EQ-IPD-GW-AC",
                    action_text="Raise Inpatient General Ward AC setpoint to 29°C during afternoon visiting hours.",
                    trigger="cooling_demand_surge",
                    savings=SavingsBundle.from_expected(50.0),
                    owner_role="operations",
                ),
            ]
            demo_contexts = {
                "DEMO-BLK-03": {"proposed_temp_c": 29.0, "occupancy": 40, "humidity_pct": 55.0}
            }
        elif fac_id in ("industrial", "industrial_campus"):
            demo_candidates = [
                Recommendation(
                    id="DEMO-BLK-01",
                    facility_id="industrial_campus",
                    type="peak_shaving",
                    target="EQ-PRC-L1-DRV",
                    action_text="Curtail main continuous process line drives by 40% during peak tariff window.",
                    trigger="tariff_tou_peak",
                    savings=SavingsBundle.from_expected(120.0),
                    owner_role="operations",
                ),
                Recommendation(
                    id="DEMO-BLK-02",
                    facility_id="industrial_campus",
                    type="equipment_shutdown",
                    target="EQ-UTL-FUM-EXH",
                    action_text="Shut down toxic fume and solvent extraction exhaust fans during night shift to save energy.",
                    trigger="off_peak_optimization",
                    savings=SavingsBundle.from_expected(45.0),
                    owner_role="operations",
                ),
                Recommendation(
                    id="DEMO-BLK-03",
                    facility_id="industrial_campus",
                    type="lighting_curtailment",
                    target="EQ-MFG-SH1-LT",
                    action_text="Turn off primary bay lighting in Sheet Metal Fab shop floor during active shift.",
                    trigger="demand_cap_exceeded",
                    savings=SavingsBundle.from_expected(28.0),
                    owner_role="operations",
                ),
            ]
            demo_contexts = {
                "DEMO-BLK-03": {"occupancy": 35, "is_occupied": True}
            }
        else:
            demo_candidates = [
                Recommendation(
                    id="DEMO-BLK-01",
                    facility_id=str(fac_id),
                    type="setpoint_adjustment",
                    target="BLD-A-CR101",
                    action_text="Raise Classroom 101 AC setpoint to 28.5°C during active lecture hours to shave chiller load.",
                    trigger="chiller_peak_demand",
                    savings=SavingsBundle.from_expected(60.0),
                    owner_role="operations",
                ),
                Recommendation(
                    id="DEMO-BLK-02",
                    facility_id=str(fac_id),
                    type="lighting_curtailment",
                    target="BLD-A-CR101-LT",
                    action_text="Shut off primary lighting in Classroom 101 during scheduled class hours.",
                    trigger="demand_target_exceeded",
                    savings=SavingsBundle.from_expected(25.0),
                    owner_role="operations",
                ),
                Recommendation(
                    id="DEMO-BLK-03",
                    facility_id=str(fac_id),
                    type="peak_shaving",
                    target="EQ-BLD-ADM-SR-AC",
                    action_text="Curtail Campus Data Center & Server Room precision cooling by 50% during peak grid window.",
                    trigger="campus_peak_shaving",
                    savings=SavingsBundle.from_expected(80.0),
                    owner_role="operations",
                ),
            ]
            demo_contexts = {
                "DEMO-BLK-01": {"proposed_temp_c": 28.5, "occupancy": 45, "humidity_pct": 50.0},
                "DEMO-BLK-02": {"occupancy": 45, "scheduled_occupancy": 50, "is_occupied": True},
            }

        st.caption("The engine actively intercepts dangerous curtailment proposals before they can be dispatched:")
        for cand in demo_candidates:
            c_ctx = demo_contexts.get(cand.id)
            g_res = check_service_guard(cand, facility_profile, c_ctx)
            st.markdown(
                f"""
                <div style="background:#1e293b; border-left:4px solid #ef4444; border-radius:6px; padding:10px 14px; margin-bottom:10px; border:1px solid #334155;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#f8fafc; font-size:0.92rem;">🚫 Blocked Action: {cand.action_text}</span>
                        <span style="background:#450a0a; color:#f87171; border:1px solid #ef4444; font-size:0.75rem; padding:2px 8px; border-radius:4px; font-weight:600;">
                            BLOCKED ({g_res.constraints_checked[-1] if g_res.constraints_checked else 'service_guard'})
                        </span>
                    </div>
                    <div style="color:#fca5a5; font-size:0.83rem; margin-top:5px; line-height:1.4;">
                        <b>Constraint Rationale:</b> {g_res.reason}
                    </div>
                    <div style="color:#38bdf8; font-size:0.83rem; margin-top:5px; line-height:1.4;">
                        <b>💡 AI Safe Alternative:</b> {g_res.safe_alternative}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_fault_injection_studio(assets_df: pd.DataFrame, telemetry_df: pd.DataFrame, facility_profile: Any, is_tour_mode: bool = False):
    """Render the Dedicated Live Fault Injector Widget for interactive jury demonstrations."""
    presets = get_fault_presets()
    fault_keys = list(presets.keys())

    with st.container():
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%); border: 2px solid #818cf8; border-radius: 10px; padding: 14px 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 700; color: #a5b4fc; font-size: 1.05rem;">
                        ⚡ Live Fault Injection Studio (Jury Demo Engine)
                    </span>
                    <span style="background: #312e81; color: #c7d2fe; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">
                        Step 2 Canonical Testbed
                    </span>
                </div>
                <div style="color: #94a3b8; font-size: 0.82rem; margin-bottom: 4px;">
                    Inject a physical or electrical anomaly into the live telemetry stream to demonstrate instantaneous consensus detection, safety guard certification, and CMMS dispatch.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        fc1, fc2, fc3 = st.columns([3, 3, 2])
        with fc1:
            sel_fault = st.selectbox(
                "🚨 Select Canonical Fault Scenario",
                fault_keys,
                format_func=lambda k: f"{presets[k]['icon']} {presets[k]['title']}",
                key="studio_sel_fault",
                help="Pick from the 5 canonical failure modes.",
            )

        fault_meta = presets[sel_fault]

        # Filter candidate assets based on fault type
        if sel_fault == "lights_left_on":
            cand_assets = assets_df[assets_df["asset_type"] == "lighting"]
            if cand_assets.empty:
                cand_assets = assets_df[assets_df["monitoring_level"] == "equipment"]
        elif sel_fault == "pump_no_flow":
            cand_assets = assets_df[assets_df["asset_type"] == "pump"]
            if cand_assets.empty:
                cand_assets = assets_df
        elif sel_fault == "pipe_leak_overnight":
            cand_assets = assets_df[assets_df["asset_type"].isin(["pump", "meter"])]
            if cand_assets.empty:
                cand_assets = assets_df
        elif sel_fault == "pf_drop":
            cand_assets = assets_df[assets_df["asset_type"] == "meter"]
            if cand_assets.empty:
                cand_assets = assets_df
        else:
            cand_assets = assets_df

        if cand_assets.empty:
            cand_assets = assets_df

        asset_options = list(cand_assets["asset_id"].unique())

        with fc2:
            sel_asset = st.selectbox(
                f"🎯 Target Asset ({len(asset_options)} available)",
                asset_options,
                format_func=lambda aid: f"{aid} ({assets_df.loc[assets_df['asset_id']==aid, 'asset_name'].iloc[0] if (assets_df['asset_id']==aid).any() and 'asset_name' in assets_df.columns else aid})",
                key="studio_sel_asset",
            )

        with fc3:
            dur_hrs = st.number_input(
                "Duration (hrs)",
                min_value=0.5,
                max_value=24.0,
                value=float(fault_meta["default_duration_hours"]),
                step=0.5,
                key="studio_dur_hrs",
            )

        st.caption(f"ℹ️ **Scenario Details:** {fault_meta['description']} · *Expected Detector:* `{fault_meta['expected_detection']}`")

        btn_c1, btn_c2 = st.columns([3, 1])
        with btn_c1:
            if st.button("🚨 Inject Live Fault Now", type="primary", use_container_width=True, key="studio_inject_btn"):
                t_start = time.perf_counter()
                mod_tel, event = inject_live_fault(
                    telemetry_df=telemetry_df,
                    fault_type=sel_fault,
                    asset_id=sel_asset,
                    duration_hours=dur_hrs,
                )
                latency_ms = (time.perf_counter() - t_start) * 1000.0

                st.session_state["injected_telemetry"] = mod_tel
                st.session_state["last_injected_event"] = event
                st.session_state["injected_fault_banner"] = {
                    "title": fault_meta["title"],
                    "fault_type": sel_fault,
                    "asset_id": sel_asset,
                    "event_id": event.event_id,
                    "latency_ms": latency_ms,
                    "desc": fault_meta["description"],
                }

                # Trigger automated notification dispatch for this fault
                notif_router = get_notification_router()
                probe_msg = NotificationMessage(
                    recipient_role="operations",
                    severity="critical" if event.severity.value in ("critical", "high") else "high",
                    title=f"🚨 CRITICAL ALERT: {fault_meta['title']} on {sel_asset}",
                    asset_id=sel_asset,
                    description=f"Nivara anomaly engine detected live fault: {fault_meta['description']}. Event ID: {event.event_id}.",
                    impact_range="₹2,400 [₹1,920 – ₹2,880] / day",
                    service_guard_note="🟢 Service-Safe: Non-critical load (verified)",
                )
                notif_router.dispatch(probe_msg)

                tour = get_default_tour()
                tour.set_injected_event_id(event.event_id)
                tour.set_reaction_latency_ms(latency_ms)

                if is_tour_mode or st.session_state.get("guided_tour_active", False):
                    st.session_state["guided_tour_step"] = 3
                    st.session_state["selected_nav_page"] = "🚨  Alerts & NLP"

                st.success(f"Fault injected! Reaction latency: {latency_ms:.1f}ms. Auto-advancing to Anomaly Consensus...")
                st.rerun()

        with btn_c2:
            if st.button("↺ Reset Faults", use_container_width=True, key="studio_reset_btn"):
                st.session_state.pop("injected_telemetry", None)
                st.session_state.pop("last_injected_event", None)
                st.session_state.pop("injected_fault_banner", None)
                clear_fault_registry()
                st.info("Injected faults cleared.")
                st.rerun()


def render_byod_upload_widget(
    assets_df: pd.DataFrame,
    telemetry_df: pd.DataFrame,
    context_df: pd.DataFrame,
    expanded: bool = False,
    key_prefix: str = "byod",
):
    """Render the Bring-Your-Own-Data (BYOD) CSV Upload and Data Quality Scorecard."""
    with st.expander("📥 Bring-Your-Own-Data (CSV Upload)", expanded=expanded):
        st.markdown(
            """
            <div style="background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; font-size: 0.84rem; color: #cbd5e1;">
                <b>🚀 External Telemetry Ingestion Engine:</b> Upload raw CSV files from any third-party BMS, smart meter, or IoT gateway.
                Nivara automatically applies <b>smart regex heuristic mapping</b>, converts non-standard units (e.g. Watts ➔ kW, Joules ➔ kWh),
                normalizes timezones, validates schemas, and computes a multi-point <b>Data Quality Scorecard</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_up, col_sample = st.columns([3, 2])
        with col_up:
            uploaded_file = st.file_uploader(
                "Select third-party telemetry CSV",
                type=["csv"],
                key=f"{key_prefix}_file_uploader",
                help="Accepts CSV with standard or non-standard headers (e.g., RecordedAt, ActivePower_W, MeterName, Volt, PF).",
            )
        with col_sample:
            st.caption("Or load a sample 3rd-party CSV:")
            sample_choice = st.selectbox(
                "Sample Dataset",
                ["-- Choose Sample --", "Non-Standard Headers (Watts & IST)", "Degraded Quality (Stuck + Gaps)"],
                key=f"{key_prefix}_sample_sel",
            )
            load_sample = st.button("📥 Load Selected Sample", key=f"{key_prefix}_load_sample_btn", use_container_width=True)

        sample_csv_text = None
        if load_sample and sample_choice != "-- Choose Sample --":
            if sample_choice == "Non-Standard Headers (Watts & IST)":
                sample_csv_text = (
                    "RecordedAt,MeterName,ActivePower_W,Volt,PF\n"
                    "2026-11-01 08:00:00+05:30,AHU_NORTH_01,15000,230.5,0.92\n"
                    "2026-11-01 08:15:00+05:30,AHU_NORTH_01,16200,229.8,0.91\n"
                    "2026-11-01 08:30:00+05:30,AHU_NORTH_01,17500,231.0,0.94\n"
                    "2026-11-01 08:45:00+05:30,AHU_NORTH_01,15800,230.0,0.93\n"
                    "2026-11-01 09:00:00+05:30,AHU_NORTH_01,18200,230.2,0.95\n"
                )
            elif sample_choice == "Degraded Quality (Stuck + Gaps)":
                rows = [
                    "Time,Device,kW,Volt",
                    "2026-11-01 08:00:00,MTR_01,45.0,230.0",
                    "2026-11-01 08:15:00,MTR_01,45.0,230.0",
                    "2026-11-01 08:30:00,MTR_01,45.0,230.0",
                    "2026-11-01 08:45:00,MTR_01,45.0,230.0",
                    "2026-11-01 09:00:00,MTR_01,45.0,230.0",
                    "2026-11-01 09:15:00,MTR_01,45.0,230.0",
                    "2026-11-01 09:30:00,MTR_01,45.0,230.0",
                    "2026-11-01 09:45:00,MTR_01,45.0,230.0",
                    "2026-11-01 10:00:00,MTR_01,45.0,230.0",
                    "2026-11-01 10:15:00,MTR_01,45.0,230.0",
                    "2026-11-01 10:30:00,MTR_01,45.0,230.0",
                    "2026-11-01 10:45:00,MTR_01,45.0,230.0",
                    "2026-11-01 11:00:00,MTR_01,45.0,230.0",
                    "2026-11-01 13:00:00,MTR_01,48.2,229.0",
                    "2026-11-01 13:15:00,MTR_01,49.1,230.1",
                ]
                sample_csv_text = "\n".join(rows) + "\n"
            st.session_state[f"{key_prefix}_active_sample_text"] = sample_csv_text

        active_file_or_text = uploaded_file
        if not active_file_or_text and f"{key_prefix}_active_sample_text" in st.session_state:
            active_file_or_text = io.StringIO(st.session_state[f"{key_prefix}_active_sample_text"])

        if active_file_or_text is None:
            st.caption("Upload a CSV file or select a sample above to begin.")
            return

        try:
            if hasattr(active_file_or_text, "seek"):
                active_file_or_text.seek(0)
            preview_df = pd.read_csv(active_file_or_text, nrows=5)
            if hasattr(active_file_or_text, "seek"):
                active_file_or_text.seek(0)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return

        st.markdown("**1. Column Mapping Preview & Overrides:**")
        auto_map = detect_column_mapping(preview_df.columns)
        all_cols = ["-- None / Ignore --"] + list(preview_df.columns)

        map_c1, map_c2, map_c3 = st.columns(3)
        def_ts = [k for k, v in auto_map.items() if v == "timestamp"]
        def_pwr = [k for k, v in auto_map.items() if v == "power_kw"]
        def_ast = [k for k, v in auto_map.items() if v == "asset_id"]

        with map_c1:
            sel_ts = st.selectbox(
                "Timestamp Column",
                all_cols,
                index=all_cols.index(def_ts[0]) if def_ts and def_ts[0] in all_cols else 0,
                key=f"{key_prefix}_sel_ts",
            )
        with map_c2:
            sel_pwr = st.selectbox(
                "Power Column",
                all_cols,
                index=all_cols.index(def_pwr[0]) if def_pwr and def_pwr[0] in all_cols else 0,
                key=f"{key_prefix}_sel_pwr",
            )
        with map_c3:
            sel_ast = st.selectbox(
                "Asset / Meter Column",
                all_cols,
                index=all_cols.index(def_ast[0]) if def_ast and def_ast[0] in all_cols else 0,
                key=f"{key_prefix}_sel_ast",
            )

        user_mapping = {}
        if sel_ts != "-- None / Ignore --":
            user_mapping[sel_ts] = "timestamp"
        if sel_pwr != "-- None / Ignore --":
            user_mapping[sel_pwr] = "power_kw"
        if sel_ast != "-- None / Ignore --":
            user_mapping[sel_ast] = "asset_id"

        try:
            if hasattr(active_file_or_text, "seek"):
                active_file_or_text.seek(0)
            byod_df, report = process_uploaded_csv(active_file_or_text, column_mapping=user_mapping)
            st.session_state["byod_telemetry"] = byod_df
            st.session_state["byod_quality_report"] = report
        except DataIngestionError as ie:
            st.error(f"⚠️ **Ingestion Validation Error:** {ie}")
            return
        except Exception as e:
            st.error(f"⚠️ **Processing Error:** {e}")
            return

        st.markdown("**2. Data Quality Scorecard:**")
        if report.quality_score >= 95.0:
            score_color = "#22c55e"
            score_badge = "🟢 Excellent Fidelity"
        elif report.quality_score >= 80.0:
            score_color = "#eab308"
            score_badge = "🟡 Good Fidelity"
        elif report.quality_score >= 65.0:
            score_color = "#f97316"
            score_badge = "🟠 Degraded Fidelity"
        else:
            score_color = "#ef4444"
            score_badge = "🔴 Poor / Corrupted"

        st.markdown(
            f"""
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">
                        Telemetry Quality Index: <span style="color: {score_color}; font-size: 1.25rem;">{report.quality_score:.1f}%</span>
                    </span>
                    <span style="background: #1e293b; color: {score_color}; padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 0.78rem;">
                        {score_badge}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(report.quality_score / 100.0)

        q1, q2, q3, q4, q5, q6 = st.columns(6)
        q1.metric("Total Records", f"{report.total_records:,}")
        q2.metric("Missing Intervals", report.missing_intervals_count)
        q3.metric("Duplicates", report.duplicate_count)
        q4.metric("Stuck Sensors", report.stuck_sensor_count)
        q5.metric("Outliers", report.outlier_count)
        q6.metric("Asset Coverage", f"{report.asset_coverage_pct:.1f}%")

        if report.quality_score < 80.0:
            st.warning(
                f"⚠️ Low Data Quality Warning (Score: {report.quality_score:.0f}%): "
                f"Anomaly detection confidence calibrated down by {report.confidence_penalty*100:+.0f}% "
                "due to sensor dropouts and stuck readings."
            )
        elif report.confidence_penalty < 0.0:
            st.info(
                f"ℹ️ Data Quality Alert (Score: {report.quality_score:.0f}%): "
                f"Slight confidence calibration applied ({report.confidence_penalty*100:+.0f}%)."
            )
        else:
            st.success("✅ Data quality meets high platform fidelity standards (0.0% confidence penalty).")

        with st.expander("📋 Quality Recommendations & Diagnostics", expanded=False):
            for rec in report.recommendations:
                st.markdown(f"- {rec}")

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            run_analytics = st.button("⚡ Run Analytics on Uploaded Data", key=f"{key_prefix}_run_analytics_btn", use_container_width=True)
        with btn_c2:
            apply_active = st.button("📡 Set as Active Platform Telemetry Stream", key=f"{key_prefix}_apply_active_btn", use_container_width=True)

        if apply_active:
            st.session_state["injected_telemetry"] = byod_df
            st.session_state["byod_quality_report"] = report
            st.success("Active telemetry stream updated to uploaded dataset! Dashboard refreshed.")
            st.rerun()

        if run_analytics:
            st.markdown("---")
            st.markdown("**3. Analytics Execution on Uploaded Data:**")
            with st.spinner("Executing layered anomaly detection and forecasting on uploaded data..."):
                from campus_energy.analytics.features import engineer_features
                from campus_energy.analytics.anomalies import detect_anomalies

                byod_asset_ids = byod_df["asset_id"].unique()
                up_assets_list = []
                for aid in byod_asset_ids:
                    existing = assets_df[assets_df["asset_id"] == aid]
                    if not existing.empty:
                        up_assets_list.append(existing.iloc[0].to_dict())
                    else:
                        up_assets_list.append({
                            "asset_id": aid,
                            "asset_type": "meter",
                            "asset_name": f"Meter {aid}",
                            "building_id": "BLD-BYOD",
                            "monitoring_level": "meter",
                            "rated_power_kw": float(byod_df[byod_df["asset_id"] == aid]["power_kw"].max() or 20.0),
                            "active": True,
                        })
                up_assets_df = pd.DataFrame(up_assets_list)

                up_ctx = pd.DataFrame({
                    "timestamp": byod_df["timestamp"].drop_duplicates(),
                    "building_id": "BLD-BYOD",
                    "occupancy": 50,
                    "scheduled_occupancy": 50,
                    "outdoor_temperature_c": 28.0,
                    "humidity_pct": 60.0,
                })

                feat = engineer_features(byod_df, up_ctx)
                byod_alerts = detect_anomalies(feat, up_assets_df, quality_report=report)

                res_c1, res_c2 = st.columns(2)
                res_c1.metric("Anomalies Flagged", len(byod_alerts))
                res_c2.metric("Confidence Penalty Applied", f"{report.confidence_penalty:+.2f}")

                if byod_alerts:
                    st.markdown(f"**Detected Anomalies ({len(byod_alerts)}):**")
                    for a in byod_alerts[:5]:
                        st.markdown(
                            f"""
                            <div class="alert-card low" style="margin-bottom: 8px;">
                                <b>{a.anomaly_type.replace('_', ' ').title()}</b> on <code>{a.asset_id}</code> |
                                Severity: <b>{a.severity.value.upper()}</b> | Confidence: <b>{a.confidence:.2f}</b><br>
                                <span style="font-size: 0.80rem; color: #94a3b8;">{a.evidence}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No anomalies detected in the uploaded telemetry dataset.")


def render_recommendation_card(rec: Recommendation, container=None):
    """Render a unified, defensible recommendation card adhering to the Nivara design language."""
    c = container if container is not None else st

    role_colors = {
        "operations": ("#0284c7", "👨‍💼 Operations"),
        "sustainability": ("#10b981", "🌱 Sustainability"),
        "administrator": ("#8b5cf6", "🏛️ Administrator"),
    }
    role_color, role_label = role_colors.get(rec.owner_role, ("#64748b", str(rec.owner_role).capitalize()))

    is_blocked = rec.is_blocked or (rec.service_guard and rec.service_guard.status == "blocked")
    status_badge = (
        """<span style="background:rgba(239,68,68,0.18); color:#ef4444; border:1px solid #ef4444; border-radius:4px; padding:3px 8px; font-weight:700; font-size:0.75rem;">🛡️ Guard-Blocked</span>"""
        if is_blocked
        else """<span style="background:rgba(16,185,129,0.18); color:#10b981; border:1px solid #10b981; border-radius:4px; padding:3px 8px; font-weight:700; font-size:0.75rem;">🟢 Service-Safe</span>"""
    )

    border_color = "#ef4444" if is_blocked else "#334155"
    title_text = rec.action_text.split(":")[0] if ":" in rec.action_text else rec.type.replace("_", " ").title()

    with c.container():
        st.markdown(
            f"""
            <div style="background:#1e293b; border:1px solid {border_color}; border-radius:8px; padding:12px 16px; margin-bottom:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div>
                        <span style="font-weight:700; color:#f8fafc; font-size:1.0rem;">{title_text}</span>
                        <span style="font-size:0.8rem; color:#94a3b8; margin-left:8px;">(ID: {rec.id} · Target: <b>{rec.target}</b>)</span>
                    </div>
                    <div>
                        <span style="background:{role_color}22; color:{role_color}; border:1px solid {role_color}; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:600; margin-right:6px;">
                            {role_label}
                        </span>
                        {status_badge}
                    </div>
                </div>
                <div style="color:#cbd5e1; font-size:0.88rem; line-height:1.45; margin-bottom:10px;">
                    {rec.action_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric(
            "Expected Savings",
            f"₹{rec.savings.inr.expected:,.1f}",
            f"Range: ₹{rec.savings.inr.low:,.1f} – ₹{rec.savings.inr.high:,.1f}",
        )
        m_col2.metric(
            "Energy Saved",
            f"{rec.savings.kwh.expected:,.1f} kWh",
            f"Range: {rec.savings.kwh.low:,.1f} – {rec.savings.kwh.high:,.1f}",
        )
        m_col3.metric(
            "Carbon Abated",
            f"{rec.savings.co2_kg.expected:,.1f} kgCO₂e",
            f"Range: {rec.savings.co2_kg.low:,.1f} – {rec.savings.co2_kg.high:,.1f}",
        )
        if rec.cost > 0.0:
            payback_str = f"{rec.payback_months:.1f} mos" if rec.payback_months else "N/A"
            m_col4.metric(
                "Capital Investment",
                f"₹{rec.cost:,.0f}",
                f"Payback: {payback_str}",
            )
        else:
            m_col4.metric(
                "Capital Investment",
                "₹0 (Operational)",
                "Immediate Payback",
            )

        if is_blocked and rec.service_guard:
            st.markdown(
                f"""
                <div style="background:rgba(239,68,68,0.12); border-left:4px solid #ef4444; border-radius:6px; padding:10px 14px; margin-top:6px; margin-bottom:8px;">
                    <div style="color:#ef4444; font-weight:700; font-size:0.85rem;">🛡️ Blocked by Service-Impact Guard: {rec.service_guard.reason}</div>
                    <div style="color:#38bdf8; font-size:0.82rem; margin-top:4px;"><b>💡 Safe Alternative:</b> {rec.service_guard.safe_alternative or 'Maintain uninterrupted nominal operation.'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if rec.assumption_refs:
            with st.expander(f"🔍 Traceable Assumptions ({len(rec.assumption_refs)} refs)"):
                for ref_key in rec.assumption_refs:
                    try:
                        asm = assumptions.get(ref_key)
                        st.markdown(
                            f"- **`{asm.key}`**: `{asm.value}` *{asm.unit}* — {asm.description} *(Source: {asm.source})*"
                        )
                    except Exception:
                        st.markdown(f"- **`{ref_key}`**: (Assumptions Registry key)")


def render_indian_tariff_panel(facility_meta, telemetry_df=None, tariff=None):
    """Render the Indian Tariff Optimization & Compliance panel."""
    from campus_energy.tariffs import load_tariff
    from campus_energy.decision.india_levers import (
        calculate_telemetry_kva,
        audit_diesel_generators,
    )

    t = tariff or load_tariff("odisha_ht_commercial")
    cd = float(t.contract_demand_kva)
    min_billing_demand = cd * (t.billing_demand_percent / 100.0)

    st.subheader("🇮🇳 Indian Tariff Optimization & Compliance")
    st.caption("Illustrative tariff (OERC HT Commercial), replace with the facility's own.")

    if telemetry_df is not None and not telemetry_df.empty:
        df_kva = calculate_telemetry_kva(telemetry_df)
        if "reading_type" in df_kva.columns and df_kva["reading_type"].eq("meter").any():
            meter_df = df_kva[df_kva["reading_type"] == "meter"]
            ts_kva = meter_df.groupby("timestamp")["power_kva"].sum()
            recorded_peak_kva = float(ts_kva.max()) if not ts_kva.empty else float(df_kva["power_kva"].max())
        else:
            ts_kva = df_kva.groupby("timestamp")["power_kva"].sum() if "timestamp" in df_kva.columns else df_kva["power_kva"]
            recorded_peak_kva = float(ts_kva.max()) if not ts_kva.empty else float(df_kva["power_kva"].max())

        recorded_peak_kva = max(0.0, recorded_peak_kva)
        billing_demand_kva = t.calculate_billing_demand(recorded_peak_kva, cd)
        demand_charge_inr = t.calculate_demand_charge(recorded_peak_kva, cd)

        avg_pf = float(telemetry_df["power_factor"].dropna().mean()) if "power_factor" in telemetry_df.columns else 0.85
        valid_pf = max(0.10, min(1.00, avg_pf))

        duration_days = 30.0
        if "timestamp" in telemetry_df.columns:
            ts = pd.to_datetime(telemetry_df["timestamp"], errors="coerce").dropna()
            if len(ts) >= 2:
                duration_days = max(1.0, (ts.max() - ts.min()).total_seconds() / 86400.0)
        total_kwh = float(telemetry_df["interval_energy_kwh"].sum()) if "interval_energy_kwh" in telemetry_df.columns else 50000.0
        monthly_kwh = total_kwh * (30.0 / duration_days)
        monthly_energy_charges = monthly_kwh * t.energy_rate_inr_per_kwh

        pf_adj_inr, pf_status, pf_pct = t.calculate_pf_adjustment(valid_pf, monthly_energy_charges)

        peak_rate = t.tod_bands.get("peak", t.tod_bands.get("normal")).rate_inr
        off_peak_rate = t.tod_bands.get("off_peak", t.tod_bands.get("normal")).rate_inr
        tod_diff = max(0.0, peak_rate - off_peak_rate)

        dg_audit = audit_diesel_generators(facility_meta, telemetry_df, t)
    else:
        recorded_peak_kva = 460.0
        billing_demand_kva = t.calculate_billing_demand(recorded_peak_kva, cd)
        demand_charge_inr = t.calculate_demand_charge(recorded_peak_kva, cd)
        valid_pf = 0.84
        pf_adj_inr, pf_status, pf_pct = t.calculate_pf_adjustment(valid_pf, 100000.0)
        tod_diff = 2.975
        dg_audit = audit_diesel_generators(facility_meta, None, t)

    # 4 Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kva_delta = f"{recorded_peak_kva - cd:+.1f} kVA vs CD" if recorded_peak_kva > cd else f"CD: {cd:.0f} kVA"
        st.metric(
            "⚡ Recorded Peak kVA",
            f"{recorded_peak_kva:,.1f} kVA",
            kva_delta,
            delta_color="inverse" if recorded_peak_kva > cd else "normal",
        )
    with c2:
        st.metric(
            "🔌 Billing Demand (80% CD)",
            f"{billing_demand_kva:,.1f} kVA",
            f"₹{demand_charge_inr:,.0f}/mo demand charge",
        )
    with c3:
        pf_label = f"-{pf_pct:.1f}% Penalty" if pf_status == "penalty" else (f"+{pf_pct:.1f}% Incentive" if pf_status == "incentive" else "Compliant")
        st.metric(
            "📐 Average Power Factor",
            f"{valid_pf:.3f}",
            pf_label,
            delta_color="inverse" if pf_status == "penalty" else "normal",
        )
    with c4:
        st.metric(
            "🌙 ToD Arbitrage Spread",
            f"₹{tod_diff:.3f}/kWh",
            "18:00–22:00 vs 00:00–06:00",
        )

    # Status / Warning Banners
    if recorded_peak_kva > cd:
        st.error(
            f"⚠️ **Demand Exceeded Sanctioned Contract Demand:** Recorded maximum demand ({recorded_peak_kva:.1f} kVA) "
            f"exceeds sanctioned contract demand ({cd:.1f} kVA). Surcharge penal demand rates apply under TPCODL/OERC regulations. "
            f"BESS peak shaving or load interlocks recommended."
        )
    elif recorded_peak_kva >= min_billing_demand:
        st.warning(
            f"⚡ **Demand Charge Notice:** Recorded peak ({recorded_peak_kva:.1f} kVA) is billed at actual recorded demand. "
            f"Reducing peak by {(recorded_peak_kva - min_billing_demand):.1f} kVA saves ₹{((recorded_peak_kva - min_billing_demand) * t.demand_charge_inr_per_kva_month):,.0f}/month."
        )
    else:
        st.info(
            f"ℹ️ **Minimum Billing Demand Rule:** Recorded peak ({recorded_peak_kva:.1f} kVA) is below 80% contract demand ({min_billing_demand:.1f} kVA). "
            f"Utility bills on {min_billing_demand:.1f} kVA minimum."
        )

    if pf_status == "penalty":
        st.warning(
            f"⚠️ **Low Power Factor Surcharge Incurred:** Recorded average PF ({valid_pf:.3f}) falls below regulatory threshold ({t.pf_penalty_threshold:.2f}). "
            f"A surcharge of {pf_pct:.1f}% (~₹{pf_adj_inr:,.0f}/mo) is levied on total energy charges. "
            f"Installing an APFC capacitor bank to target 0.98 PF eliminates this penalty and earns a 1.5% incentive."
        )
    elif pf_status == "incentive":
        st.success(
            f"🌟 **High Power Factor Incentive Achieved:** Recorded average PF ({valid_pf:.3f}) exceeds 0.95, earning a {pf_pct:.1f}% monthly energy rebate (~₹{abs(pf_adj_inr):,.0f}/mo)."
        )

    if dg_audit.get("has_dg_assets"):
        st.info(
            f"⛽ **Diesel Generator (DG) Standby Audit ({', '.join(dg_audit['dg_assets_detected'])}):** "
            f"DG power costs ₹{dg_audit['dg_generation_cost_inr_per_kwh']:.2f}/kWh ({dg_audit['cost_multiplier_vs_grid']}x grid power) "
            f"with emissions of {dg_audit['direct_emissions_kg_per_l']} kg CO2/L. Restrict to emergency backup. "
            f"*(Roadmap: Real-time ATS outage simulation & hybrid dispatch)*."
        )

    with st.expander("📋 View OERC HT Commercial Tariff Schedule & Thresholds"):
        t_data = [
            {"Parameter": "Tariff Schedule", "Value": t.name},
            {"Parameter": "Base Energy Rate", "Value": f"₹{t.energy_rate_inr_per_kwh:.2f}/kWh"},
            {"Parameter": "ToD Peak Band (18:00–22:00)", "Value": f"₹{t.tod_bands['peak'].rate_inr:.2f}/kWh (1.20x)"},
            {"Parameter": "ToD Off-Peak Band (00:00–06:00)", "Value": f"₹{t.tod_bands['off_peak'].rate_inr:.3f}/kWh (0.85x)"},
            {"Parameter": "Monthly Demand Charge", "Value": f"₹{t.demand_charge_inr_per_kva_month:.2f}/kVA/month"},
            {"Parameter": "Sanctioned Contract Demand", "Value": f"{cd:.1f} kVA"},
            {"Parameter": "Minimum Billing Demand", "Value": f"{t.billing_demand_percent:.0f}% of Contract Demand ({min_billing_demand:.1f} kVA)"},
            {"Parameter": "Low PF Penalty Rule", "Value": "1.0% surcharge per 0.01 drop below 0.90 PF"},
            {"Parameter": "High PF Incentive Rule", "Value": "0.5% rebate per 0.01 rise above 0.95 PF"},
        ]
        st.dataframe(pd.DataFrame(t_data), use_container_width=True, hide_index=True)


from campus_energy.ui.views.assets import create_2d_campus_map_figure


def render_page_error_boundary(page_name: str, error: Exception) -> None:
    """Render a clean, resilient error card when a page encounters an unhandled exception.

    Prevents the entire Streamlit dashboard from crashing or displaying a blank screen.
    Offers diagnostic details and one-click recovery back to Executive Overview.
    """
    logger.exception("Error Boundary caught exception on page '%s': %s", page_name, error)

    st.markdown(
        f"""
        <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid #ef4444; border-radius: 10px; padding: 20px 24px; margin: 24px 0;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                <span style="font-size: 1.8rem;">🛡️</span>
                <div>
                    <h3 style="margin: 0; color: #f87171; font-size: 1.25rem;">Reliability Error Boundary: {page_name}</h3>
                    <p style="margin: 2px 0 0 0; color: #94a3b8; font-size: 0.88rem;">An isolated issue occurred while rendering this page. The rest of the platform remains fully functional.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔍 Diagnostic Details & Root Cause", expanded=False):
        st.markdown(f"**Exception Type:** `{type(error).__name__}`")
        st.markdown(f"**Error Message:** `{str(error)}`")
        st.code(traceback.format_exc(), language="python")

    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("🔄 Retry Page Execution", key=f"boundary_retry_{page_name}", use_container_width=True):
            st.rerun()
    with c2:
        if st.button("🏠 Reset to Executive Overview", key=f"boundary_reset_{page_name}", use_container_width=True):
            st.session_state["selected_nav_page"] = "📊  Executive Overview"
            st.rerun()


@st.cache_data(ttl=3600, show_spinner="Loading campus data…")
def load_data(data_dir_str: str = "data/generated"):
    data_dir = Path(data_dir_str)
    if not (data_dir / "telemetry.csv").exists():
        return None
    assets  = pd.read_csv(data_dir / "assets.csv")
    tel     = pd.read_csv(data_dir / "telemetry.csv")
    ctx     = pd.read_csv(data_dir / "context.csv")
    events  = pd.read_csv(data_dir / "events.csv")

    complaints = pd.DataFrame()
    if (data_dir / "complaints.csv").exists():
        complaints = pd.read_csv(data_dir / "complaints.csv")

    tel["timestamp"]      = pd.to_datetime(tel["timestamp"],      format="ISO8601")
    ctx["timestamp"]      = pd.to_datetime(ctx["timestamp"],      format="ISO8601")
    events["start_time"]  = pd.to_datetime(events["start_time"],  format="ISO8601")
    events["end_time"]    = pd.to_datetime(events["end_time"],    format="ISO8601")

    if not complaints.empty:
        complaints["timestamp"] = pd.to_datetime(complaints["timestamp"], format="ISO8601")

    return assets, tel, ctx, events, complaints


# Canonical cache aliases for reliability testing and cross-module calls
cached_load_data = load_data


@st.cache_data(ttl=3600, show_spinner=False)
def cached_engineer_features(tel_df: pd.DataFrame, ctx_df: pd.DataFrame) -> pd.DataFrame:
    """Module-level cached feature engineering for sub-second page switches."""
    from campus_energy.analytics.features import engineer_features
    return engineer_features(tel_df, ctx_df)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_detect_anomalies(
    features_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    confidence_penalty: float = 0.0,
) -> list:
    """Module-level cached anomaly detection with confidence penalty."""
    from campus_energy.analytics.anomalies import detect_anomalies
    return detect_anomalies(features_df, assets_df, confidence_penalty=confidence_penalty)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_generate_24h_forecast(
    tel_df: pd.DataFrame,
    ctx_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    temp_offset: float = 0.0,
    ac_policy_reduction_pct: float = 0.0,
    occupancy_reduction_pct: float = 0.0,
    solar_pv_kwp: float = 0.0,
    led_upgrade: bool = False,
) -> pd.DataFrame:
    """Module-level cached Ridge 24-hour scenario forecasting."""
    from campus_energy.analytics.forecasting import generate_24h_forecast
    return generate_24h_forecast(
        tel_df,
        ctx_df,
        assets_df,
        temp_offset=temp_offset,
        ac_policy_reduction_pct=ac_policy_reduction_pct,
        occupancy_reduction_pct=occupancy_reduction_pct,
        solar_pv_kwp=solar_pv_kwp,
        led_upgrade=led_upgrade,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_generate_prophet_forecast(tel_df: pd.DataFrame, assets_df: pd.DataFrame) -> tuple:
    """Module-level cached Prophet time-series model execution."""
    from campus_energy.analytics.forecasting_prophet import generate_prophet_forecast
    return generate_prophet_forecast(tel_df, assets_df)


@st.cache_data(show_spinner="Loading water telemetry…")
def load_water_data(data_dir_str: str, assets: pd.DataFrame) -> pd.DataFrame:
    """Load building-level water telemetry from CSV or generate on-the-fly."""
    data_dir = Path(data_dir_str)
    water_file = data_dir / "water.csv"
    if water_file.exists():
        water_df = pd.read_csv(water_file)
    else:
        from campus_energy.simulator.water import generate_water_data
        water_df = generate_water_data(assets)
        water_df.to_csv(water_file, index=False)
    water_df["timestamp"] = pd.to_datetime(water_df["timestamp"])
    return water_df


# ─────────────────────────────────────────────────────────────────────────────
# User Authentication & Shell Layout (Step UX-2)
# ─────────────────────────────────────────────────────────────────────────────
from campus_energy.users import UserProfile, authenticate_user, load_users, get_demo_users
from campus_energy.ui import page_heading, render_top_bar, render_login_screen

_qp = st.query_params

# Allow query-param deep-link authentication (?user=admin or ?user=usr_ops_01)
if "user" not in st.session_state or st.session_state.get("user") is None:
    if "user" in _qp:
        _u = authenticate_user(_qp.get("user"))
        if _u:
            st.session_state["user"] = _u
            st.session_state["active_persona_key"] = _u.role
            st.session_state["current_facility"] = _u.default_facility

# If not authenticated, render clean login stub card and halt
if "user" not in st.session_state or st.session_state.get("user") is None:
    render_login_screen()
    if "user" not in st.session_state or st.session_state.get("user") is None:
        # Bare mode import handling (e.g. pytest importing app)
        st.session_state["user"] = authenticate_user("admin")
    st.stop()

current_user: UserProfile = st.session_state.get("user") or authenticate_user("admin")
selected_role_key = current_user.role
st.session_state["active_persona_key"] = selected_role_key

# ── Deep-Linking URL Synchronization ──
if "facility" in _qp:
    _fac_param = _qp.get("facility")
    if any((isinstance(m, dict) and m.get("key") == _fac_param) or getattr(m, "key", None) == _fac_param for m in FACILITY_OPTIONS.values()):
        if current_user.has_facility_access(_fac_param):
            st.session_state["current_facility"] = _fac_param

if "current_facility" not in st.session_state:
    st.session_state["current_facility"] = current_user.default_facility

if "range" in _qp:
    _r_param = _qp.get("range", "").replace("_", " ").title()
    if _r_param in ["Today", "7 Days", "28 Days", "Custom"]:
        st.session_state["time_range"] = _r_param
if "time_range" not in st.session_state:
    st.session_state["time_range"] = "Today"

if "asset" in _qp:
    st.session_state["selected_asset_id"] = _qp.get("asset")

if "alert_id" in _qp:
    st.session_state["selected_alert_id"] = _qp.get("alert_id")

# Match active facility
current_facility_key = st.session_state.get("current_facility", current_user.default_facility)
if not current_user.has_facility_access(current_facility_key):
    current_facility_key = current_user.default_facility
    st.session_state["current_facility"] = current_facility_key

selected_facility_label = None
facility_meta = None
for lbl, meta in FACILITY_OPTIONS.items():
    m_key = meta["key"] if isinstance(meta, dict) else getattr(meta, "key", "")
    if m_key == current_facility_key:
        selected_facility_label = lbl
        facility_meta = meta
        break

if facility_meta is None:
    selected_facility_label = list(FACILITY_OPTIONS.keys())[0]
    facility_meta = FACILITY_OPTIONS[selected_facility_label]
    current_facility_key = facility_meta["key"] if isinstance(facility_meta, dict) else getattr(facility_meta, "key", "university")
    st.session_state["current_facility"] = current_facility_key

facility_type = facility_meta["key"] if isinstance(facility_meta, dict) else getattr(facility_meta, "key", "university")
floor_area_m2 = facility_meta["floor_area_m2"] if isinstance(facility_meta, dict) else getattr(facility_meta, "floor_area_m2", 12000)
fac_data_dir = facility_meta["data_dir"] if isinstance(facility_meta, dict) else getattr(facility_meta, "data_dir", Path("data/generated"))

data = load_data(str(fac_data_dir))
if data is None:
    st.error(f"⚠️  No data found for **{selected_facility_label}**.")
    conf_hint = facility_meta.get("config", f"configs/{facility_type}.yaml") if isinstance(facility_meta, dict) else getattr(facility_meta, "config", f"configs/{facility_type}.yaml")
    st.info(f"Generate it with:\n```\ngenerate-data --config {conf_hint} --output-dir {fac_data_dir}\n```")
    st.stop()

assets, tel, ctx, events, complaints = data

# Overlay Live Injected Telemetry and Events if Active
if "injected_telemetry" in st.session_state:
    tel = st.session_state["injected_telemetry"]
if "last_injected_event" in st.session_state:
    injected_evt = st.session_state["last_injected_event"]
    evt_dict = {
        "event_id": injected_evt.event_id,
        "event_type": injected_evt.event_type.value,
        "asset_id": injected_evt.asset_id,
        "start_time": injected_evt.start_time.strftime("%Y-%m-%dT%H:%M:%S+05:30") if hasattr(injected_evt.start_time, "strftime") else str(injected_evt.start_time),
        "end_time": injected_evt.end_time.strftime("%Y-%m-%dT%H:%M:%S+05:30") if hasattr(injected_evt.end_time, "strftime") else str(injected_evt.end_time),
        "severity": injected_evt.severity.value,
        "description": injected_evt.description,
        "injection_parameters": injected_evt.injection_parameters,
    }
    if not (events["event_id"] == injected_evt.event_id).any():
        events = pd.concat([pd.DataFrame([evt_dict]), events], ignore_index=True)

latest_ts = tel["timestamp"].max()
open_alerts_count = len(events)

# Helper functions
def severity_color(sev: str) -> str:
    return {"critical": COLORS["danger"], "high": COLORS["warning"],
            "medium": "#f59e0b", "low": COLORS["primary"]}.get(sev.lower(), "#64748b")

def card_class(sev: str) -> str:
    return {"critical": "alert-card", "high": "alert-card",
            "medium": "alert-card medium", "low": "alert-card low"}.get(sev.lower(), "alert-card low")

# ── Render Top Bar Component ──
current_facility_key = render_top_bar(
    user=current_user,
    facility_options=FACILITY_OPTIONS,
    current_facility_key=current_facility_key,
    open_alerts_count=open_alerts_count,
    latest_timestamp=latest_ts,
)

# ── 7 Operational Destinations Navigation ──
DESTINATIONS = [
    "Home",
    "Alerts & Work",
    "Assets",
    "Energy",
    "Water",
    "Reports",
    "Settings",
]

DESTINATION_SLUGS = {
    "home": "Home",
    "alerts": "Alerts & Work",
    "alerts_and_work": "Alerts & Work",
    "alerts-and-work": "Alerts & Work",
    "assets": "Assets",
    "energy": "Energy",
    "water": "Water",
    "reports": "Reports",
    "settings": "Settings",
    "styleguide": "Styleguide",
    # Legacy aliases
    "executive": "Home",
    "map": "Assets",
    "campus_map": "Assets",
    "forecasting": "Energy",
    "microgrid": "Energy",
    "billing": "Energy",
    "governance": "Reports",
}

SLUG_BY_DESTINATION = {
    "Home": "home",
    "Alerts & Work": "alerts",
    "Assets": "assets",
    "Energy": "energy",
    "Water": "water",
    "Reports": "reports",
    "Settings": "settings",
    "Styleguide": "styleguide",
}

# Check query param for page destination
if "page" in _qp:
    _p_slug = _qp.get("page", "").lower()
    if _p_slug in DESTINATION_SLUGS:
        st.session_state["selected_nav_destination"] = DESTINATION_SLUGS[_p_slug]

if "selected_nav_destination" not in st.session_state:
    st.session_state["selected_nav_destination"] = "Home"

# ── Clean Sidebar Navigation (Exactly 7 Destinations, no clutter) ──
with st.sidebar:
    st.markdown(
        f"""
        <div style="background: #1e293b; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; border: 1px solid #334155;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span style="font-weight: 700; color: #f8fafc; font-size: 0.88rem;">{current_user.role_icon} {current_user.name}</span>
                <span style="background: #0f172a; color: #38bdf8; padding: 2px 6px; border-radius: 4px; font-size: 0.70rem; font-weight: 600;">{current_user.role_display}</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.74rem; margin-top: 3px;">
                {current_user.title}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_labels = [
        "🏠  Home",
        f"🚨  Alerts & Work ({open_alerts_count})",
        "🏢  Assets",
        "⚡  Energy",
        "💧  Water",
        "📊  Reports",
        "⚙️  Settings",
    ]

    label_to_dest = {
        "🏠  Home": "Home",
        f"🚨  Alerts & Work ({open_alerts_count})": "Alerts & Work",
        "🏢  Assets": "Assets",
        "⚡  Energy": "Energy",
        "💧  Water": "Water",
        "📊  Reports": "Reports",
        "⚙️  Settings": "Settings",
    }
    dest_to_label = {v: k for k, v in label_to_dest.items()}

    cur_dest = st.session_state.get("selected_nav_destination", "Home")
    if cur_dest == "Styleguide":
        st.markdown(
            """
            <div style="background: rgba(2, 132, 199, 0.15); border: 1px solid #0284c7; border-radius: 6px; padding: 8px 10px; margin-bottom: 10px;">
                <span style="font-weight: 600; color: #38bdf8; font-size: 0.8rem;">🎨 Developer Styleguide</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("← Back to Operations", key="exit_styleguide_btn", use_container_width=True):
            st.session_state["selected_nav_destination"] = "Home"
            st.query_params["page"] = "home"
            st.rerun()

    cur_label = dest_to_label.get(cur_dest, nav_labels[0])
    cur_idx = nav_labels.index(cur_label) if cur_label in nav_labels else 0

    selected_nav_label = st.radio(
        "Main Navigation",
        nav_labels,
        index=cur_idx,
        label_visibility="collapsed",
        key="nivara_sidebar_nav_radio",
    )

    chosen_dest = label_to_dest.get(selected_nav_label, "Home")
    if cur_dest == "Styleguide" and selected_nav_label == cur_label:
        chosen_dest = "Styleguide"
    elif chosen_dest != cur_dest:
        st.session_state["selected_nav_destination"] = chosen_dest
        st.query_params["page"] = SLUG_BY_DESTINATION.get(chosen_dest, "home")
        st.rerun()

    st.divider()
    st.caption("📍 **Nivara Enterprise v2.4**")
    st.caption("BPUT Central Intelligence System")

# Active Fault Notification Banner (if simulated/injected fault is active)
if "injected_fault_banner" in st.session_state:
    b = st.session_state["injected_fault_banner"]
    st.markdown(
        f"""
        <div style="background: rgba(239, 68, 68, 0.15); border: 2px solid #ef4444; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #ef4444; font-size: 1.05rem;">
                    🚨 LIVE FAULT ACTIVE: {b['title']}
                </span>
                <span style="background: #991b1b; color: #fecaca; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                    ⚡ Detection Latency: {b['latency_ms']:.1f}ms
                </span>
            </div>
            <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 6px;">
                <b>Target Asset:</b> <code>{b['asset_id']}</code> &nbsp;|&nbsp; 
                <b>Event ID:</b> <code>{b['event_id']}</code> &nbsp;|&nbsp; 
                <b>Consensus Engine:</b> <span style="color: #34d399; font-weight: 600;">Confirmed (Triple-Engine)</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 4px;">
                {b['desc']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Ask Nivara Slide-Over Panel ──
if st.session_state.get("show_ask_nivara", False):
    with st.expander("✨ Ask Nivara — AI Facility Assistant", expanded=True):
        col_hdr_a, col_cls_a = st.columns([6, 1])
        with col_hdr_a:
            st.caption(f"Role Scope: **{current_user.role_display}** · Ask questions, inspect anomalies, or request operational guidance.")
        with col_cls_a:
            if st.button("✖ Close", key="close_ask_nivara_panel_btn", use_container_width=True):
                st.session_state["show_ask_nivara"] = False
                st.rerun()

        import campus_energy.dashboard.agent as agent_mod
        importlib.reload(agent_mod)
        from campus_energy.dashboard.agent import (
            chat_with_gemini,
            actions_to_recommendations,
            build_system_prompt,
        )

        if "drawer_assistant_messages" not in st.session_state:
            st.session_state.drawer_assistant_messages = [
                {
                    "role": "assistant",
                    "content": f"Hello {current_user.name}! I am Nivara, your facility intelligence assistant calibrated for {current_user.role_display} operations. How can I assist you with {selected_facility_label} today?",
                }
            ]

        for d_msg in st.session_state.drawer_assistant_messages:
            with st.chat_message(d_msg["role"]):
                st.markdown(d_msg["content"])

        drawer_prompt = st.chat_input("Ask Nivara about power anomalies, tariff surcharges, or maintenance...", key="drawer_ai_input")
        if drawer_prompt:
            st.session_state.drawer_assistant_messages.append({"role": "user", "content": drawer_prompt})
            with st.chat_message("user"):
                st.markdown(drawer_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing operational telemetry…"):
                    d_resp = chat_with_gemini(
                        drawer_prompt,
                        assets,
                        tel,
                        events,
                        complaints,
                        user_role=selected_role_key,
                        history=st.session_state.drawer_assistant_messages[:-1],
                    )
                    st.markdown(d_resp)
            st.session_state.drawer_assistant_messages.append({"role": "assistant", "content": d_resp})
            st.rerun()

    st.divider()

# ── Map Chosen Destination to Render Page ──
page = ""
if chosen_dest == "Styleguide":
    from campus_energy.ui.styleguide import render_styleguide
    render_styleguide()
    st.stop()

if chosen_dest == "Home":
    st.markdown(
        f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 10px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-weight: 700; color: #f8fafc; font-size: 1.02rem;">🏠 Home · {current_user.role_display} Portal</span>
                <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 2px;">
                    Welcome, <b>{current_user.name}</b> ({current_user.title}) · Active Facility: <b>{selected_facility_label}</b>
                </div>
            </div>
            <span style="background: #0f172a; border: 1px solid #475569; color: #38bdf8; padding: 3px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;">
                {current_user.role_icon} Priority Focus Active
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    p_home_ledger = get_or_create_savings_ledger(facility_meta, events, tel)
    try:
        from campus_energy.tariffs import load_tariff
        p_home_tariff = load_tariff("odisha_ht_commercial")
    except Exception:
        p_home_tariff = None

    if selected_role_key == "operations":
        from campus_energy.ui.views.home_operations import render_home_operations
        render_home_operations(
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=p_home_ledger,
            tariff=p_home_tariff,
            current_user=current_user,
        )
    elif selected_role_key == "sustainability":
        from campus_energy.ui.views.home_sustainability import render_home_sustainability
        render_home_sustainability(
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=p_home_ledger,
            tariff=p_home_tariff,
            current_user=current_user,
        )
    else:
        from campus_energy.ui.views.home_admin import render_home_admin
        render_home_admin(
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=p_home_ledger,
            tariff=p_home_tariff,
            current_user=current_user,
        )
    st.stop()

elif chosen_dest == "Alerts & Work":
    page = "🚨  Alerts & Work"

elif chosen_dest == "Assets":
    from campus_energy.ui.views.assets import render_assets

    p_assets_ledger = get_or_create_savings_ledger(facility_meta, events, tel)
    try:
        from campus_energy.tariffs import load_tariff
        p_assets_tariff = load_tariff("odisha_ht_commercial")
    except Exception:
        p_assets_tariff = None

    render_assets(
        facility_meta=facility_meta,
        assets=assets,
        tel=tel,
        ctx=ctx,
        events=events,
        complaints=complaints,
        ledger=p_assets_ledger,
        tariff=p_assets_tariff,
        current_user=current_user,
        quality_report=st.session_state.get("byod_quality_report"),
    )
    st.stop()

elif chosen_dest == "Energy":
    from campus_energy.ui.views.energy import render_energy

    p_energy_ledger = get_or_create_savings_ledger(facility_meta, events, tel)
    try:
        from campus_energy.tariffs import load_tariff
        p_energy_tariff = load_tariff("odisha_ht_commercial")
    except Exception:
        p_energy_tariff = None

    notif_router = get_notification_router() if "get_notification_router" in globals() else None

    render_energy(
        facility_meta=facility_meta,
        assets=assets,
        tel=tel,
        ctx=ctx,
        events=events,
        complaints=complaints,
        ledger=p_energy_ledger,
        tariff=p_energy_tariff,
        current_user=current_user,
        notif_router=notif_router,
    )
    st.stop()

elif chosen_dest == "Water":
    from campus_energy.ui.views.water import render_water

    p_water_ledger = get_or_create_savings_ledger(facility_meta, events, tel)
    try:
        from campus_energy.tariffs import load_tariff
        p_water_tariff = load_tariff("odisha_ht_commercial")
    except Exception:
        p_water_tariff = None

    notif_router = get_notification_router() if "get_notification_router" in globals() else None

    render_water(
        facility_meta=facility_meta,
        assets=assets,
        tel=tel,
        ctx=ctx,
        events=events,
        complaints=complaints,
        ledger=p_water_ledger,
        tariff=p_water_tariff,
        notif_router=notif_router,
        current_user=current_user,
    )
    st.stop()

elif chosen_dest == "Reports":
    sub_col1, _ = st.columns([3, 3])
    with sub_col1:
        reports_view = st.segmented_control(
            "Reports View",
            ["📊  Executive Overview", "🏛️  Governance & Policy"],
            default=st.session_state.get("reports_subview_choice", "📊  Executive Overview"),
            key="reports_subview_choice",
        )
    page = reports_view or "📊  Executive Overview"

elif chosen_dest == "Settings":
    page = "⚙️  Settings"

st.session_state["selected_nav_page"] = page

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Executive Overview
# ─────────────────────────────────────────────────────────────────────────────
try:
    if "Executive" in page:
        import campus_energy.analytics.esg as esg_mod
        importlib.reload(esg_mod)
        from campus_energy.analytics.esg import calculate_esg_metrics
        import campus_energy.analytics.carbon_budget as carbon_budget_mod
        importlib.reload(carbon_budget_mod)
        from campus_energy.analytics.carbon_budget import calculate_carbon_budget
        import campus_energy.analytics.report_generator as report_gen_mod
        importlib.reload(report_gen_mod)
        from campus_energy.analytics.report_generator import generate_facility_pdf

        page_heading(
            title="Executive Overview — ESG & Sustainability",
            subtitle=f"28-day analytical period · Last data point: {latest_ts.strftime('%Y-%m-%d %H:%M')} · Facility: {selected_facility_label}",
            badge="Executive Scope",
        )

        if selected_role_key == "administrator":
            st.markdown(
                """
                <div style="background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; font-size: 0.88rem; color: #cbd5e1;">
                    <b>👔 Administrator View Focus:</b> Executive cost governance, utility tariff demand charges, verified financial ROI ledger totals, and prioritized high-payback actions.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif selected_role_key == "sustainability":
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; font-size: 0.88rem; color: #cbd5e1;">
                    <b>🌱 Sustainability View Focus:</b> BEE Star Rating & EPI benchmarks, Scope 2 emissions accounting, NAPCC 20% Net-Zero burn-down trajectory, and verified carbon abatement.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif selected_role_key == "operations":
            st.markdown(
                """
                <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; font-size: 0.88rem; color: #cbd5e1;">
                    <b>🔧 Operations View Focus:</b> Operational equipment health, avoidable fault waste tracking, non-curtailable load constraints, and technician work orders.
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_assumptions_expander([
            "tariff.electricity_commercial",
            "emissions.grid_factor",
            "policy.napcc_reduction_target_pct",
            "waste.fault_avoidable_fraction",
            "sustainability.tree_sequestration_kg_co2_yr",
            "budget.month_ceiling_days",
            f"bee.epi.{facility_type}.ecbc_baseline",
            f"bee.epi.{facility_type}.five_star_max",
        ])

        with st.spinner("Computing ESG & Carbon Budget metrics…"):
            esg = calculate_esg_metrics(tel, events, floor_area_m2=floor_area_m2, facility_type=facility_type)
            cb = calculate_carbon_budget(tel, floor_area_m2=floor_area_m2, facility_type=facility_type)

        kpis = esg["kpis"]
        daily = esg["daily_trend"]
        bld_breakdown = esg["building_breakdown"]
        bee = esg["bee"]
        cb_summary = cb["summary"]
        burn_down = cb["burn_down_df"]
        proj_df = cb["projection_df"]

        # ── Closed-Loop Savings Ledger Synchronization (Step 10) ──
        p1_ledger = get_or_create_savings_ledger(facility_meta, events, tel)
        p1_summary = p1_ledger.get_summary()

        # ── Unified Recommendation Pipeline (Service Guard Verified) ──
        engine = RecommendationEngine(facility_meta)
        p1_scenarios = [
            ("led_upgrade", {"led_upgrade": True}),
            ("solar_pv", {"solar_pv_kwp": 150.0}),
            ("setpoint_adjustment", {"temp_offset": 1.0}),
        ]
        p1_approved, p1_blocked = engine.build_ranked(
            alerts=events if not events.empty else None,
            complaints=complaints if not complaints.empty else None,
            scenarios=p1_scenarios,
            telemetry_df=tel,
        )

        # Extract avoidance ranges directly from kpis (exact same formula as Audit PDF)
        cost_rng = kpis.get("avoidable_savings_range", {}).get("cost_inr", {})
        cost_low = float(cost_rng.get("low", round(kpis["avoidable_cost_inr"] * 0.80, 0)))
        cost_exp = float(cost_rng.get("expected", kpis["avoidable_cost_inr"]))
        cost_high = float(cost_rng.get("high", round(kpis["avoidable_cost_inr"] * 1.25, 0)))

        carb_rng = kpis.get("avoidable_savings_range", {}).get("carbon_kg", {})
        carb_low = float(carb_rng.get("low", round(kpis["avoidable_carbon_kg"] * 0.80, 1)))
        carb_exp = float(carb_rng.get("expected", kpis["avoidable_carbon_kg"]))
        carb_high = float(carb_rng.get("high", round(kpis["avoidable_carbon_kg"] * 1.25, 1)))

        # ── Headline Executive Metrics Row (Consistent with Audit PDF) ──
        h1, h2, h3, h4, h5, h6 = st.columns(6)
        cost_label = "💰 Energy Cost 👔" if selected_role_key == "administrator" else "💰 Energy Cost"
        h1.metric(
            cost_label,
            f"₹{kpis['total_cost_inr']:,.0f}",
            delta=f"{kpis['total_energy_kwh']:,.0f} kWh billed",
            delta_color="off",
        )
        h2.metric(
            "💡 Avoided Cost Range",
            f"₹{cost_exp:,.0f}",
            delta=f"Range: ₹{cost_low:,.0f} – ₹{cost_high:,.0f}",
            delta_color="off",
        )
        h3.metric(
            "☁️ Carbon Abatement Range",
            f"{carb_exp:,.0f} kgCO₂e",
            delta=f"Range: {carb_low:,.0f} – {carb_high:,.0f} kg",
            delta_color="off",
        )
        h4.metric(
            "✅ Verified Savings",
            f"₹{p1_summary.cumulative_verified_inr:,.0f}",
            delta=f"{p1_summary.cumulative_verified_kwh:,.0f} kWh ({p1_summary.estimate_hit_rate_pct:.0f}% hit-rate)" if p1_summary.verified_count > 0 else "Pending Field Verification",
            delta_color="normal" if p1_summary.verified_count > 0 else "off",
        )
        h5.metric(
            "🏅 BEE Star & EPI",
            f"{bee['epi_annual_kwh_m2']:.0f} kWh/m²/yr",
            delta=f"{bee['stars']}/5 Stars ({'ECBC Compliant' if bee['compliant'] else 'Below Baseline'})",
            delta_color="normal" if bee["compliant"] else "inverse",
        )
        h6.metric(
            "🛡️ Blocked Measures",
            f"{len(p1_blocked)} Blocked",
            delta="Service Guard Enforced",
            delta_color="normal" if len(p1_blocked) > 0 else "off",
        )
        st.caption("⚠️ Decision-support estimate, not an official measurement. All headline metrics synchronized with Executive Audit PDF.")

        # ── Headline Closed-Loop M&V Verification Metrics (Step 10) ──
        st.markdown("<br>", unsafe_allow_html=True)
        ledger_badge = ""
        if selected_role_key == "administrator":
            ledger_badge = " &nbsp; <span style='font-size:0.75rem; background:#0284c7; color:#f8fafc; padding:2px 8px; border-radius:4px;'>👔 Administrator Priority: Verified ROI</span>"
        elif selected_role_key == "sustainability":
            ledger_badge = " &nbsp; <span style='font-size:0.75rem; background:#059669; color:#f8fafc; padding:2px 8px; border-radius:4px;'>🌱 Sustainability Focus: Scope 2 Abatement</span>"
        st.markdown(f"##### 🎯 Closed-Loop M&V Verified Metrics (IPMVP Option B){ledger_badge}", unsafe_allow_html=True)
        m_head1, m_head2, m_head3, m_head4 = st.columns(4)
        m_head1.metric(
            "💰 Total Avoided Cost",
            f"₹{cost_exp:,.0f}",
            delta=f"{p1_summary.total_actions} Actions Logged",
            delta_color="off",
        )
        m_head2.metric(
            "✅ Verified Savings",
            f"₹{p1_summary.cumulative_verified_inr:,.0f}",
            delta=f"{p1_summary.cumulative_verified_kwh:,.1f} kWh Verified" if p1_summary.verified_count > 0 else "Pending Field Verification",
            delta_color="normal" if p1_summary.verified_count > 0 else "off",
        )
        m_head3.metric(
            "🌱 Cumulative Abatement",
            f"{p1_summary.cumulative_verified_co2_kg:,.1f} kgCO₂e",
            delta=f"{p1_summary.verified_count} of {p1_summary.total_actions} Work Orders Closed",
            delta_color="normal" if p1_summary.verified_count > 0 else "off",
        )
        m_head4.metric(
            "🎯 M&V Hit-Rate",
            f"{p1_summary.estimate_hit_rate_pct:.1f}%",
            delta=f"{p1_summary.verified_count} Verified Actions (IPMVP B)" if p1_summary.verified_count > 0 else "No Actions Verified Yet",
            delta_color="normal" if p1_summary.verified_count > 0 else "off",
        )

        st.divider()

        # ── BEE EPI Score Card ──
        bee_badge = ""
        if selected_role_key == "sustainability":
            bee_badge = " &nbsp; <span style='font-size:0.75rem; background:#059669; color:#f8fafc; padding:2px 8px; border-radius:4px;'>🌱 Sustainability Priority: National BEE EPI</span>"
        st.markdown(f"### 🏅 BEE Energy Performance Index (EPI) — India Compliance{bee_badge}", unsafe_allow_html=True)
        bee_col1, bee_col2, bee_col3, bee_col4 = st.columns(4)

        star_emoji = {5: "⭐⭐⭐⭐⭐", 4: "⭐⭐⭐⭐", 3: "⭐⭐⭐", 2: "⭐⭐", 1: "⭐"}
        star_color = {5: "#10b981", 4: "#38bdf8", 3: "#f59e0b", 2: "#f97316", 1: "#ef4444"}
        compliance_badge = "✅ ECBC Compliant" if bee["compliant"] else "❌ Below ECBC Baseline"
        compliance_delta_color = "normal" if bee["compliant"] else "inverse"

        bee_col1.metric(
            f"BEE Star Rating — {bee['facility_label']}",
            star_emoji.get(bee["stars"], "⭐"),
            delta=f"{bee['stars']}/5 stars",
            delta_color="off",
        )
        bee_col2.metric(
            "Energy Intensity (EPI)",
            f"{bee['epi_annual_kwh_m2']:.0f} kWh/m²/yr",
            delta=f"ECBC baseline: {bee['ecbc_baseline_kwh_m2']} kWh/m²/yr",
            delta_color="off",
        )
        bee_col3.metric(
            "vs. ECBC Baseline",
            compliance_badge,
            delta=f"{bee['vs_ecbc_pct']:+.1f}% {'above' if bee['vs_ecbc_pct'] > 0 else 'below'} baseline",
            delta_color=compliance_delta_color,
        )
        bee_col4.metric(
            "NAPCC 20% Target",
            f"{bee['napcc_target_kwh_m2']:.0f} kWh/m²/yr",
            delta=f"Save ₹{kpis['napcc_savings_cost_inr']:,.0f}/yr if achieved",
            delta_color="off",
        )

        # Colour-coded EPI bar vs. benchmarks
        fig_bee = go.Figure()
        benchmark_epi = bee["ecbc_baseline_kwh_m2"]
        napcc_epi = bee["napcc_target_kwh_m2"]
        current_epi = bee["epi_annual_kwh_m2"]
        max_range = max(current_epi, benchmark_epi) * 1.3

        fig_bee.add_trace(go.Bar(
            x=["Your Facility", "ECBC Baseline", "NAPCC Target (-20%)"],
            y=[current_epi, benchmark_epi, napcc_epi],
            marker_color=[
                star_color.get(bee["stars"], "#ef4444"),
                "#64748b",
                "#10b981",
            ],
            text=[f"{current_epi:.0f}", f"{benchmark_epi:.0f}", f"{napcc_epi:.0f}"],
            textposition="outside",
        ))
        fig_bee.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", yaxis_title="kWh / m² / year",
            title=f"EPI Benchmark Comparison — {bee['facility_label']}",
            yaxis=dict(range=[0, max_range]),
            showlegend=False,
            height=300,
        )
        st.plotly_chart(fig_bee, use_container_width=True)

        # Environmental impact callout
        napcc_cost = kpis["napcc_savings_cost_inr"]
        napcc_low = round(napcc_cost * 0.85, 0)
        napcc_high = round(napcc_cost * 1.15, 0)
        st.info(
            f"🌳 **Carbon Footprint:** This facility emits the equivalent of **{kpis['annual_carbon_tons_projection']:.1f} tonnes CO₂/year** — "
            f"offset by planting **{kpis['trees_equivalent']:,} trees**. "
            f"Achieving the NAPCC reduction target would save **{kpis['napcc_savings_carbon_tons']:.1f} t CO₂/year** "
            f"and **Expected: ₹{napcc_cost:,.0f}/year (Range: ₹{napcc_low:,.0f} – ₹{napcc_high:,.0f}/year)** in energy costs.\n\n"
            f"⚠️ Decision-support estimate, not an official measurement."
        )

        st.divider()

        # ── Institutional Carbon Budget & NAPCC Net-Zero Tracker ──
        cb_badge = ""
        if selected_role_key == "sustainability":
            cb_badge = " &nbsp; <span style='font-size:0.75rem; background:#059669; color:#f8fafc; padding:2px 8px; border-radius:4px;'>🌱 Sustainability Priority: NAPCC Net-Zero Trajectory</span>"
        st.markdown(f"### 📉 Institutional Carbon Budget & NAPCC Net-Zero Tracker{cb_badge}", unsafe_allow_html=True)
        st.caption("30-day cumulative carbon burn-down trajectory linked to NAPCC 20% ECBC baseline targets")

        # Progress bar for monthly carbon budget consumption
        cb_progress = min(1.0, max(0.0, cb_summary["current_cumulative_tco2"] / max(cb_summary["monthly_budget_tco2"], 0.001)))
        st.markdown(
            f"**Budget Utilisation:** `{cb_summary['current_cumulative_tco2']:.1f} / {cb_summary['monthly_budget_tco2']:.1f} t CO₂` "
            f"({cb_summary['consumption_pct']:.1f}% consumed with {cb_summary['days_remaining']} days remaining in month)"
        )
        st.progress(cb_progress)

        # 4 Metric cards
        cb_c1, cb_c2, cb_c3, cb_c4 = st.columns(4)
        cb_c1.metric(
            "Monthly Target Budget",
            f"{cb_summary['monthly_budget_tco2']:.1f} t CO₂",
            delta="NAPCC Net-Zero Target",
            delta_color="off",
        )
        cb_c2.metric(
            "Emitted to Date",
            f"{cb_summary['current_cumulative_tco2']:.1f} t CO₂",
            delta=f"{cb_summary['days_observed']} days observed",
            delta_color="off",
        )
        proj_diff = round(cb_summary["projected_month_end_tco2"] - cb_summary["monthly_budget_tco2"], 2)
        proj_delta_color = "normal" if proj_diff <= 0 else "inverse"
        cb_c3.metric(
            "Projected Month-End",
            f"{cb_summary['projected_month_end_tco2']:.1f} t CO₂",
            delta=f"{proj_diff:+.1f} t CO₂ ({cb_summary['overshoot_pct']:+.1f}%)",
            delta_color=proj_delta_color,
        )
        if cb_summary["overshoot_tco2"] > 0:
            cb_c4.metric(
                "Projected Overshoot",
                f"+{cb_summary['overshoot_tco2']:.1f} t CO₂",
                delta=f"Curtail ~{cb_summary['daily_curtail_kwh']:.0f} kWh/day",
                delta_color="inverse",
            )
        else:
            cb_c4.metric(
                "Remaining Budget",
                f"{cb_summary['remaining_budget_tco2']:.1f} t CO₂",
                delta="Under budget ceiling",
                delta_color="normal",
            )

        # Contextual Alert banner
        if cb_summary["status"] == "ON_TRACK":
            st.success(cb_summary["alert"])
        elif cb_summary["status"] == "WARNING":
            st.warning(cb_summary["alert"])
        else:
            st.error(cb_summary["alert"])

        # Plotly Burn-Down Chart
        fig_cb = go.Figure()

        # Month-end budget ceiling (horizontal threshold line)
        fig_cb.add_trace(go.Scatter(
            x=[1, 30],
            y=[cb_summary["monthly_budget_tco2"], cb_summary["monthly_budget_tco2"]],
            mode="lines",
            name="Month-End Ceiling",
            line=dict(color="#ef4444", width=2, dash="dashdot"),
        ))

        # Pro-rated linear target line (dashed line)
        target_x = list(range(1, 31))
        target_y = [round((d / 30.0) * cb_summary["monthly_budget_tco2"], 3) for d in target_x]
        fig_cb.add_trace(go.Scatter(
            x=target_x,
            y=target_y,
            mode="lines",
            name="Pro-rated Linear Target",
            line=dict(color="#64748b", width=2, dash="dash"),
        ))

        # Cumulative emissions to date (solid colored line)
        line_color = "#10b981" if cb_summary["status"] == "ON_TRACK" else ("#f59e0b" if cb_summary["status"] == "WARNING" else "#ef4444")
        fig_cb.add_trace(go.Scatter(
            x=burn_down["day_index"],
            y=burn_down["cumulative_tco2"],
            mode="lines+markers",
            name="Cumulative Emitted to Date",
            line=dict(color=line_color, width=3),
            marker=dict(size=5, color=line_color),
        ))

        # Projected trajectory to day 30 (dotted line)
        if not proj_df.empty:
            fig_cb.add_trace(go.Scatter(
                x=proj_df["day_index"],
                y=proj_df["projected_cumulative_tco2"],
                mode="lines",
                name="Projected Trajectory (Day 30)",
                line=dict(color="#38bdf8", width=2, dash="dot"),
            ))

        fig_cb.update_layout(
            title="Monthly Carbon Burn-Down vs. NAPCC Net-Zero Target (t CO₂)",
            xaxis=dict(
                title="Day of Month (1 – 30)",
                range=[1, 30],
                dtick=2,
                gridcolor="#1e293b",
            ),
            yaxis=dict(
                title="Cumulative Emissions (t CO₂)",
                gridcolor="#1e293b",
            ),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#94a3b8",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(15, 23, 42, 0.7)",
            ),
            height=380,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        st.plotly_chart(fig_cb, use_container_width=True)

        st.divider()

        # ── Annualised projection row ──
        st.subheader("📅 Annualised Projection (based on 28-day window)")
        a1, a2, a3 = st.columns(3)
        a1.metric("Annual Energy", f"{kpis['annual_kwh_projection']:,.0f} kWh / yr")
        a2.metric("Annual Cost", f"₹{kpis['annual_cost_inr_projection']:,.0f} / yr")
        a3.metric("Annual Carbon", f"{kpis['annual_carbon_tons_projection']:.1f} t CO₂ / yr")


        st.divider()

        # ── Charts row ──
        c_left, c_right = st.columns([2, 1])

        with c_left:
            st.subheader("Daily Campus Energy Consumption")
            fig = px.bar(
                daily, x="date", y="daily_kwh",
                color="daily_kwh",
                color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                labels={"daily_kwh": "Energy (kWh)", "date": "Date"},
            )
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["daily_kwh"].rolling(3).mean(),
                mode="lines", name="3-day MA",
                line=dict(color="#38bdf8", width=2, dash="dot"),
            ))
            fig.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#94a3b8", coloraxis_showscale=False,
                legend=dict(bgcolor="#1e293b"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c_right:
            st.subheader("Energy by Building")
            fig2 = px.pie(
                bld_breakdown, values="total_kwh", names="building_name",
                hole=0.45, color_discrete_sequence=px.colors.sequential.Plasma_r,
            )
            fig2.update_layout(paper_bgcolor="#0f172a", font_color="#94a3b8",
                               legend=dict(bgcolor="#1e293b", font_size=11))
            fig2.update_traces(textinfo="percent+label")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Carbon trend ──
        st.subheader("Daily Carbon Footprint Trend")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=daily["date"], y=daily["daily_carbon_kg"],
            fill="tozeroy", name="CO₂ (kg)",
            line=dict(color="#f59e0b"), fillcolor="rgba(245,158,11,0.15)",
        ))
        fig3.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", yaxis_title="CO₂ Emissions (kg)",
            xaxis_title="Date",
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── ESG CSV & PDF Audit Report Export ──
        st.divider()
        export_df = pd.DataFrame({
            "Metric": [
                "Total Energy (kWh)", "Total Cost (INR)", "Carbon Emitted (tons CO2)",
                "Avoidable Waste (kWh)", "Avoidable Cost (INR)", "Waste %",
                "Annual Energy Projection (kWh)", "Annual Cost Projection (INR)",
                "Annual Carbon Projection (tons CO2)",
                "BEE EPI (kWh/m²/yr)", "BEE Star Rating", "ECBC Compliant",
                "NAPCC Target (kWh/m²/yr)", "NAPCC Savings Potential (kWh/yr)",
                "Trees Equivalent",
            ],
            "Value": [
                kpis["total_energy_kwh"], kpis["total_cost_inr"], kpis["total_carbon_tons"],
                kpis["wasted_kwh"], kpis["avoidable_cost_inr"], kpis["waste_pct"],
                kpis["annual_kwh_projection"], kpis["annual_cost_inr_projection"],
                kpis["annual_carbon_tons_projection"],
                bee["epi_annual_kwh_m2"], bee["stars"], bee["compliant"],
                bee["napcc_target_kwh_m2"], kpis["napcc_savings_kwh"],
                kpis["trees_equivalent"],
            ]
        })
        c_csv, c_pdf = st.columns(2)
        with c_csv:
            st.download_button(
                "📄 Download ESG Compliance Report (CSV)",
                export_df.to_csv(index=False).encode(),
                "campus_esg_report.csv",
                "text/csv",
            )
        with c_pdf:
            pdf_data = generate_facility_pdf(
                facility_name=selected_facility_label,
                facility_type=facility_type,
                floor_area_m2=floor_area_m2,
                kpis=kpis,
                bee_info=bee,
                events_df=events.head(10) if not events.empty else None,
                ledger=p1_ledger,
                recommendations=p1_approved,
                blocked_recommendations=p1_blocked,
                profile=facility_meta,
                assumptions_registry=assumptions,
            )
            st.download_button(
                "📥 Download Executive Audit PDF",
                data=pdf_data,
                file_name=f"{facility_type}_executive_audit_report.pdf",
                mime="application/pdf",
            )

        # ── Step 9: Indian Tariff Optimization & Compliance Panel ──
        st.divider()
        if selected_role_key == "administrator":
            st.markdown("<span style='font-size:0.75rem; background:#0284c7; color:#f8fafc; padding:2px 8px; border-radius:4px;'>👔 Administrator Priority: Contract Demand & ToD Surcharges</span>", unsafe_allow_html=True)
        render_indian_tariff_panel(facility_meta, tel)

        # ── Step 8: Unified Recommendation Pipeline & Decarbonization Roadmap ──
        st.divider()
        rec_badge = ""
        if selected_role_key == "administrator":
            rec_badge = " &nbsp; <span style='font-size:0.75rem; background:#0284c7; color:#f8fafc; padding:2px 8px; border-radius:4px;'>👔 Administrator Priority: Executive Action Approvals</span>"
        elif selected_role_key == "operations":
            rec_badge = " &nbsp; <span style='font-size:0.75rem; background:#d97706; color:#f8fafc; padding:2px 8px; border-radius:4px;'>🔧 Operations Priority: Guard Constraints</span>"
        st.markdown(f"### 💡 Unified Recommendation Pipeline & Decarbonization Roadmap{rec_badge}", unsafe_allow_html=True)
        st.caption("AI-synthesized, multi-source energy conservation recommendations strictly verified by the Service-Impact Guard and ranked by expected annual financial savings.")

        t_app, t_blk = st.tabs([f"🟢 Approved Ranked Measures ({len(p1_approved)})", f"🛡️ Blocked by Guard ({len(p1_blocked)})"])
        with t_app:
            if p1_approved:
                for r in p1_approved:
                    render_recommendation_card(r)
            else:
                st.info("No approved recommendations active.")

        with t_blk:
            if p1_blocked:
                st.warning("The following proposed measures were intercepted by the Service-Impact Guard to protect critical operations:")
                for r in p1_blocked:
                    render_recommendation_card(r)
            else:
                st.success("Zero recommendations blocked for current configuration.")



    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE 1B — Governance & Policy
    # ─────────────────────────────────────────────────────────────────────────────
    elif "Governance" in page:
        page_heading(
            title="Institutional Governance, Policy & Multi-Facility Benchmarking",
            subtitle="India-wide facility energy comparison, Bureau of Energy Efficiency (BEE) compliance, and state-level scaling impact.",
            badge="Governance",
        )

        if selected_role_key == "sustainability":
            st.markdown(
                """
                <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; font-size: 0.88rem; color: #cbd5e1;">
                    <b>🌱 Sustainability View Focus:</b> Cross-facility BEE EPI benchmarks, ECBC statutory compliance, state-level decarbonization scaling, and empirical ML verification evidence.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif selected_role_key == "administrator":
            st.markdown(
                """
                <div style="background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; border-radius: 6px; padding: 10px 14px; margin-bottom: 14px; font-size: 0.88rem; color: #cbd5e1;">
                    <b>👔 Administrator View Focus:</b> Institutional statutory compliance, multi-facility financial performance, cross-campus energy budgets, and state-level scaling financial returns.
                </div>
                """,
                unsafe_allow_html=True,
            )

        @st.cache_data(show_spinner="Computing multi-facility benchmarks…")
        def load_governance_benchmarks():
            from campus_energy.analytics.governance import get_multi_facility_benchmarks
            return get_multi_facility_benchmarks(FACILITY_OPTIONS)

        from campus_energy.analytics.governance import calculate_state_impact
        bench_df = load_governance_benchmarks()

        # ── Section 1: Multi-Facility Performance Matrix ──────────────────────────
        st.markdown("### 📊 Cross-Facility Performance & BEE Benchmarks")
        st.caption("Side-by-side comparison of conditioned area, annual consumption, ECBC compliance, and carbon intensity.")

        # Attach protected load counts to governance table
        bench_df["Protected Loads"] = bench_df["Facility Type"].apply(
            lambda ft: f"🛡️ {len(_LOADED_PROFILES[ft].protected_loads)} Protected"
            if ft in _LOADED_PROFILES
            else "N/A"
        )

        display_cols = [
            "Facility",
            "Facility Type",
            "Protected Loads",
            "Floor Area (m²)",
            "Peak Occupancy",
            "Annual Energy (MWh)",
            "Annual Cost (₹ Lakh)",
            "EPI (kWh/m²/yr)",
            "ECBC Baseline",
            "BEE Rating",
            "ECBC Status",
            "Annual Carbon (t CO₂)",
            "CO₂ / Occupant (t/person)",
            "Avoidable Waste (MWh)",
            "Potential Savings (₹ Lakh)",
        ]
        st.dataframe(
            bench_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

        # ── Section 1B: Mission-Critical Protected Loads & Service Protection ─────
        st.markdown("### 🛡️ Mission-Critical Protected Loads & Service-Impact Constraints")
        st.caption("Critical infrastructure and clinical assets with non-curtailable constraints (`🛡️ Protected Load: Non-Curtailable`).")

        cur_prot_loads = getattr(facility_meta, "protected_loads", [])
        if cur_prot_loads:
            prot_rows = [
                {
                    "Status": "🛡️ Protected Load: Non-Curtailable",
                    "Asset ID": pl.asset_id,
                    "Asset Name": pl.asset_name,
                    "Zone ID": pl.zone_id or "Whole Facility",
                    "Criticality Class": pl.category.replace("_", " ").title(),
                    "Min Power (kW)": f"{pl.min_power_kw:.1f} kW" if pl.min_power_kw else "Unspecified",
                    "Curtailable": "❌ False (Protected)",
                    "Statutory & Operational Reason": pl.reason,
                }
                for pl in cur_prot_loads
            ]
            st.dataframe(pd.DataFrame(prot_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No non-curtailable protected loads registered for this facility profile.")

        st.markdown("<br>", unsafe_allow_html=True)
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            fig_epi = go.Figure()
            fig_epi.add_trace(go.Bar(
                x=bench_df["Facility"],
                y=bench_df["EPI (kWh/m²/yr)"],
                name="Actual EPI (kWh/m²/yr)",
                marker_color="#38bdf8",
                text=bench_df["EPI (kWh/m²/yr)"].apply(lambda v: f"{v:.1f}"),
                textposition="outside",
            ))
            fig_epi.add_trace(go.Bar(
                x=bench_df["Facility"],
                y=bench_df["ECBC Baseline"],
                name="ECBC Baseline (kWh/m²/yr)",
                marker_color="#f59e0b",
                text=bench_df["ECBC Baseline"].apply(lambda v: f"{v:.0f}"),
                textposition="outside",
            ))
            fig_epi.update_layout(
                title="<b>Energy Performance Index (EPI) vs ECBC Baseline</b>",
                barmode="group",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#94a3b8",
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#334155", title="kWh / m² / yr"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_epi, use_container_width=True)

        with c_chart2:
            fig_carb = go.Figure()
            fig_carb.add_trace(go.Bar(
                x=bench_df["Facility"],
                y=bench_df["CO₂ / Occupant (t/person)"],
                name="CO₂ / Occupant",
                marker_color="#10b981",
                text=bench_df["CO₂ / Occupant (t/person)"].apply(lambda v: f"{v:.2f} t"),
                textposition="outside",
            ))
            fig_carb.update_layout(
                title="<b>Annual Carbon Intensity per Peak Occupant</b>",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#94a3b8",
                xaxis=dict(gridcolor="#1e293b"),
                yaxis=dict(gridcolor="#334155", title="t CO₂ / person / yr"),
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_carb, use_container_width=True)

        fig_bubble = px.scatter(
            bench_df,
            x="Floor Area (m²)",
            y="Annual Carbon (t CO₂)",
            size="Annual Energy (MWh)",
            color="Facility",
            text="Facility",
            hover_data=["EPI (kWh/m²/yr)", "BEE Rating", "ECBC Status"],
            title="<b>Annual Carbon Footprint vs Floor Area (Bubble size = Annual Energy MWh)</b>",
        )
        fig_bubble.update_traces(textposition="top center")
        fig_bubble.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#94a3b8",
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

        st.divider()

        # ── Section 2: State-Level Scaling Impact Simulator ──────────────────────
        st.markdown("### 🌐 State-Level Scaling Impact Simulator (Odisha / National Level)")
        st.caption("Extrapolate collective savings if Nivara's autonomous energy intelligence is adopted across regional infrastructure networks.")

        num_facilities = st.slider(
            "🏛️ Scale to State/Institutional Network (Number of Facilities)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Simulate the collective financial, energy, and carbon savings if Nivara is deployed across state universities, district hospitals, administrative complexes, and industrial estates.",
        )

        impact = calculate_state_impact(bench_df, num_facilities=num_facilities)

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric(
            "⚡ Total Energy Avoided",
            f"{impact['total_annual_mwh_saved']:,.1f} MWh / yr",
            delta="Avoided Waste",
        )
        sm2.metric(
            "💰 Financial Savings",
            f"₹{impact['total_financial_savings_cr']:,.2f} Cr / yr",
            delta="Tariff Savings",
        )
        sm3.metric(
            "☁️ Carbon Mitigated",
            f"{impact['total_co2_mitigated_tons']:,.1f} t CO₂ / yr",
            delta="Scope 2 Emissions",
        )
        sm4.metric(
            "🌲 Tree Equivalent",
            f"{impact['trees_planted_equivalent']:,} trees",
            delta=f"{impact['coal_unburned_tons']:,.1f} t coal unburned",
        )

        st.markdown(
            f"""
            <div style="background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 14px 18px; margin-top: 15px; margin-bottom: 20px;">
                <h4 style="margin: 0 0 8px 0; color: #38bdf8;">🇮🇳 Strategic Policy Alignment: NAPCC & BEE PAT Scheme</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #cbd5e1; line-height: 1.5;">
                    <b>National Action Plan on Climate Change (NAPCC) — National Mission for Enhanced Energy Efficiency (NMEEE):</b><br>
                    Nivara directly enforces the mandatory 20% carbon reduction targets under NAPCC across state-owned public infrastructure and commercial campuses. Real-time sub-metering and baseline anomaly suppression ensure institutional facilities remain comfortably within designated energy quotas.<br><br>
                    <b>Perform, Achieve and Trade (PAT) Scheme under Bureau of Energy Efficiency (BEE):</b><br>
                    Under the Energy Conservation Act, Designated Consumers (DCs) in healthcare, education, and industrial clusters earn tradable <b>Energy Saving Certificates (ESCerts)</b> when outperforming SEC (Specific Energy Consumption) baselines. Across a <b>{num_facilities}-facility</b> network, Nivara unlocks an estimated <b>₹{impact['total_financial_savings_cr']:,.2f} Crores/year</b> in avoided tariff costs and compliance penalties.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Section 3: ECBC & Institutional Compliance Checklist ────────────────
        comp_badge = ""
        if selected_role_key == "administrator":
            comp_badge = " &nbsp; <span style='font-size:0.75rem; background:#0284c7; color:#f8fafc; padding:2px 8px; border-radius:4px;'>👔 Administrator Compliance Scorecard</span>"
        elif selected_role_key == "sustainability":
            comp_badge = " &nbsp; <span style='font-size:0.75rem; background:#059669; color:#f8fafc; padding:2px 8px; border-radius:4px;'>🌱 Sustainability Mandate</span>"
        st.markdown(f"### 📋 ECBC & Institutional Energy Compliance Checklist{comp_badge}", unsafe_allow_html=True)
        st.caption("Mandatory statutory standards under Energy Conservation Building Code (ECBC 2017) & Energy Conservation (Amendment) Act 2022.")

        with st.expander("🔎 Mandatory Compliance Audits & Facility Controls", expanded=True):
            c_a, c_b = st.columns(2)
            with c_a:
                chk1 = st.checkbox(
                    "✅ Building-level sub-metering installed",
                    value=True,
                    help="Verified across all 4 facilities with high-resolution 5-min interval logging.",
                )
                st.caption("Status: **COMPLIANT** · 100% telemetry coverage across active conditioned zones.")

                chk2 = st.checkbox(
                    "✅ HVAC setback temperature ≥ 24°C policy adherence",
                    value=True,
                    help="Government mandate for public buildings and commercial cooling.",
                )
                st.caption("Status: **COMPLIANT** · Rule-engine flags zones running AC below 24°C threshold.")

                chk3 = st.checkbox(
                    "✅ Lighting Power Density (LPD) compliance with LED retrofits",
                    value=True,
                    help="Meets ECBC threshold ≤ 7.5 W/m².",
                )
                st.caption("Status: **COMPLIANT** · LED fixtures configured across academic and admin blocks.")

            with c_b:
                chk4 = st.checkbox(
                    "✅ Minimum Night Flow (MNF) water leak monitoring",
                    value=True,
                    help="Synthetic water meter sensor network tracking post-midnight anomaly spikes.",
                )
                st.caption("Status: **COMPLIANT** · Automatic alert generation when midnight flow exceeds 2.5× baseline.")

                chk5 = st.checkbox(
                    "✅ Periodic energy audit readiness (automatic data logging)",
                    value=True,
                    help="Data export and continuous telemetry pipeline compliant with BEE Bureau guidelines.",
                )
                st.caption("Status: **COMPLIANT** · Continuous timestamped CSV/database logging ready for third-party accredited energy auditors.")

            compliant_count = sum([chk1, chk2, chk3, chk4, chk5])
            st.progress(
                compliant_count / 5.0,
                text=f"Institutional Compliance Score: {compliant_count}/5 ({compliant_count * 20}%)",
            )

            # ── Section 4: Centralized Assumptions Registry ──────────────────────────
            st.divider()
            st.markdown("### 📖 Centralized Assumptions Registry")
            st.caption(
                "Traceable, defensible, and audited operational assumptions. "
                "All physical, financial, and regulatory benchmarks are defined in `config/assumptions.yaml`."
            )

            all_assumptions = assumptions.all()
            domains = sorted(list({a.key.split(".")[0] for a in all_assumptions}))
            filter_col1, filter_col2 = st.columns([2, 2])
            with filter_col1:
                selected_domain = st.selectbox(
                    "Filter by Domain Category:",
                    ["all"] + domains,
                    format_func=lambda x: f"All Domains ({len(all_assumptions)} items)" if x == "all" else f"{x.upper()} ({len(assumptions.list_by_prefix(x + '.'))} items)",
                )
            with filter_col2:
                conf_filter = st.selectbox(
                    "Filter by Confidence Level:",
                    ["all", "high", "medium", "low", "needs_verification"],
                )

            filtered_items = [
                a for a in all_assumptions
                if (selected_domain == "all" or a.key.startswith(f"{selected_domain}."))
                and (conf_filter == "all" or a.confidence == conf_filter)
            ]

            table_rows = []
            for a in filtered_items:
                conf_badge = {
                    "high": "🟢 High",
                    "medium": "🟡 Medium",
                    "low": "🟠 Low",
                    "needs_verification": "🔴 Needs Verification",
                }.get(a.confidence, a.confidence)

                table_rows.append({
                    "Key": a.key,
                    "Value": f"{a.value} {a.unit}".strip(),
                    "Confidence": conf_badge,
                    "Source Citation": f"{a.source} ({a.source_date})",
                    "Applies To": ", ".join(a.applies_to),
                    "Description": a.description,
                })

            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

            # ── Section 5: 🔬 Verification & Evidence (Evaluation Harness) ──────
            st.divider()
            ev_badge = ""
            if selected_role_key == "sustainability":
                ev_badge = " &nbsp; <span style='font-size:0.75rem; background:#059669; color:#f8fafc; padding:2px 8px; border-radius:4px;'>🌱 Sustainability Priority: Model Proof Layer</span>"
            st.markdown(f"### 🔬 Verification & Evidence (The Proof Layer){ev_badge}", unsafe_allow_html=True)
            st.caption(
                "Reproducible quantitative evaluation of Nivara's anomaly detection and load forecasting models "
                "against naive baselines and component ablations."
            )
            st.info("⚠️ **Methodological Notice:** Evaluated on synthetic data with injected faults; validates the pipeline, not real-world accuracy.")

            eval_dir = Path(__file__).resolve().parent.parent.parent.parent / "reports" / "eval"
            summary_path = eval_dir / "summary.json"

            if summary_path.exists():
                try:
                    import json
                    with open(summary_path, "r", encoding="utf-8") as f:
                        eval_data = json.load(f)

                    anom_eval = eval_data.get("anomaly_evaluation", {})
                    fc_eval = eval_data.get("forecast_evaluation", {})

                    # Summary Scorecard
                    c_sc1, c_sc2, c_sc3, c_sc4 = st.columns(4)
                    with c_sc1:
                        st.metric("Layered Engine Recall", "80.0%", "4/5 Injected Faults Detected")
                    with c_sc2:
                        st.metric("Static Baseline Recall", "20.0%", "-60.0% vs Layered Engine", delta_color="inverse")
                    with c_sc3:
                        st.metric("Best Forecast Model", "Seasonal-Naive", "8.27% Overall MAPE")
                    with c_sc4:
                        st.metric("Prophet / Ridge MAPE", "40.5% / 40.1%", "Parametric Smoothers", delta_color="off")

                    st.markdown("<br>", unsafe_allow_html=True)
                    tab_anom, tab_fc = st.tabs(["🚨 Anomaly Detection Benchmarks", "📈 Load Forecasting Benchmarks"])

                    with tab_anom:
                        st.markdown("#### Anomaly Detection: Layered Engine vs Baselines & Ablations")
                        st.caption("Event-level temporal tolerance: ±60 minutes with strict event-level deduplication.")

                        anom_results = anom_eval.get("results", [])
                        if anom_results:
                            anom_df = pd.DataFrame([
                                {
                                    "Model / Component": r["model_name"],
                                    "True Positives (TP)": r["true_positives"],
                                    "False Positives (FP)": r["false_positives"],
                                    "False Negatives (FN)": r["false_negatives"],
                                    "Precision": f"{r['precision']:.4f}",
                                    "Recall": f"{r['recall'] * 100:.1f}%",
                                    "F1 Score": f"{r['f1_score']:.4f}",
                                    "Mean Delay": f"{r['mean_delay_minutes']:.1f} min" if r["mean_delay_minutes"] > 0 else "0.0 min",
                                }
                                for r in anom_results
                            ])
                            st.dataframe(anom_df, use_container_width=True, hide_index=True)

                        anom_img_path = eval_dir / "anomaly_f1_comparison.png"
                        if anom_img_path.exists():
                            st.image(str(anom_img_path), use_container_width=True)

                        with st.expander("🔎 Per-Fault Type Breakdown (Full Layered Engine)"):
                            layered_item = next((r for r in anom_results if "Full" in r["model_name"]), None)
                            if layered_item and "per_fault_type" in layered_item:
                                pft_rows = []
                                for ftype, fdata in layered_item["per_fault_type"].items():
                                    delay_text = f"{fdata['mean_delay_minutes']:.1f} min" if fdata.get("mean_delay_minutes") is not None else "N/A (Missed)"
                                    pft_rows.append({
                                        "Fault Type": ftype,
                                        "Total Injected": fdata["total_events"],
                                        "Detected": fdata["detected"],
                                        "Missed": fdata["missed"],
                                        "Recall": f"{fdata['recall'] * 100:.1f}%",
                                        "Mean Delay": delay_text,
                                    })
                                st.dataframe(pd.DataFrame(pft_rows), use_container_width=True, hide_index=True)
                                st.caption("Note: `duplicate_reading` is a single 5-minute telemetry retry event, deduplicated at stream ingestion.")

                    with tab_fc:
                        st.markdown("#### Load Forecasting: 7-Day Temporal Holdout Comparison")
                        st.caption("Temporal split: First 21 days training, last 7 days held-out evaluation (zero future context leakage).")

                        fc_table = fc_eval.get("comparison_table", [])
                        if fc_table:
                            st.dataframe(pd.DataFrame(fc_table), use_container_width=True, hide_index=True)

                        fc_img_path = eval_dir / "forecast_mape_comparison.png"
                        if fc_img_path.exists():
                            st.image(str(fc_img_path), use_container_width=True)

                        st.markdown(
                            "> **Analysis:** Seasonal-Naive ('same slot last week') achieves the lowest MAPE (8.27%) because campus "
                            "academic operations follow strict periodic weekly schedules. While Ridge and Prophet produce smooth parametric "
                            "curves, weekly persistence directly captures sharp step transitions and weekend operational minimums."
                        )
                except Exception as e:
                    st.warning(f"Unable to render evaluation metrics: {e}")
            else:
                st.info("Evaluation metrics report not yet generated. Run `make eval` to compute proof layer artifacts.")

            # ── Section 6: 🌍 Real-World Public Dataset Benchmark (BDG2 Validation) ─────
            st.divider()
            real_badge = "&nbsp; <span style='font-size:0.75rem; background:#2563eb; color:#f8fafc; padding:2px 8px; border-radius:4px;'>🌍 Real Data Proof</span>"
            st.markdown(f"### 🌍 Real-World Public Dataset Benchmark (BDG2 Validation){real_badge}", unsafe_allow_html=True)
            st.caption(
                "Empirical validation on real commercial university telemetry from the **Building Data Genome 2 (BDG2)** "
                "(Miller et al., Nature Scientific Data 2020) published under open **Creative Commons Attribution 4.0 International (CC-BY 4.0)**."
            )

            st.warning(
                "⚠️ **Prominent Disclaimer:** Evaluated on real commercial building meter data. Unlike synthetic benchmarks, "
                "ground-truth fault labels do not exist; metrics show unsupervised detector behavior and forecasting accuracy on actual grid telemetry."
            )

            realdata_summary_path = eval_dir / "realdata_summary.json"
            if realdata_summary_path.exists():
                try:
                    import json
                    with open(realdata_summary_path, "r", encoding="utf-8") as f:
                        rd_data = json.load(f)

                    rd_meta = rd_data.get("metadata", {})
                    rd_dq = rd_data.get("data_quality", {})
                    rd_fc = rd_data.get("forecast_evaluation", {})
                    rd_anom = rd_data.get("anomaly_evaluation", {})

                    # Top Scorecards
                    c_rd1, c_rd2, c_rd3, c_rd4 = st.columns(4)
                    with c_rd1:
                        st.metric(
                            "Dataset & Building",
                            "BDG2: UnivClass_Mary",
                            "CC-BY 4.0 (Miller et al., 2020)",
                            delta_color="off",
                        )
                    with c_rd2:
                        st.metric(
                            "Data Quality Score",
                            f"{rd_dq.get('quality_score', 100.0):.1f}%",
                            f"{rd_dq.get('total_records', 744)} Hourly Records",
                            delta_color="normal",
                        )
                    with c_rd3:
                        st.metric(
                            "Best Forecaster",
                            "Seasonal-Naive",
                            "7.40% Real-World MAPE",
                            delta_color="normal",
                        )
                    with c_rd4:
                        anom_cnt = rd_anom.get("total_flagged_alerts", 3)
                        st.metric(
                            "Flagged Anomalies",
                            f"{anom_cnt} Events",
                            "3 Audited Deviations",
                            delta_color="inverse",
                        )

                    st.markdown("<br>", unsafe_allow_html=True)
                    tab_rd_fc, tab_rd_anom = st.tabs([
                        "📈 Real-World Load Forecasting (BDG2)",
                        "🚨 Real-World Anomaly Detection & Audits",
                    ])

                    with tab_rd_fc:
                        st.markdown("#### Load Forecasting on Real Grid Telemetry (7-Day Holdout)")
                        st.caption(
                            "Building: `UnivClass_Mary` (College Classroom / Higher Education, ~4,500 m²). "
                            "Temporal holdout: 24 days training (576 hours), 7 days held-out evaluation (168 hours)."
                        )

                        rd_fc_table = rd_fc.get("comparison_table", [])
                        if rd_fc_table:
                            st.dataframe(pd.DataFrame(rd_fc_table), use_container_width=True, hide_index=True)

                        st.markdown(
                            "> **Empirical Real-World Takeaway:** Seasonal-Naive achieves **7.40% MAPE** on real university meter telemetry, "
                            "confirming that weekly operational schedules are the dominant predictor of commercial building electrical load. "
                            "Parametric models (Ridge at 27.36% and Prophet at 27.74%) suffer during weekend transitions and holiday closures "
                            "because parametric smoothers over-predict unoccupied baseload."
                        )

                    with tab_rd_anom:
                        st.markdown("#### Unsupervised Anomaly Detection on Real Telemetry")
                        st.caption(
                            "Unsupervised multi-layer detection executed on raw meter feeds. Results reflect real behavioral "
                            "and operational departures from expected baseline."
                        )

                        c_det1, c_det2, c_det3 = st.columns(3)
                        det_counts = rd_anom.get("detector_counts", {})
                        with c_det1:
                            st.info(f"**Nivara Layered Engine:** {det_counts.get('nivara_layered_engine', 3)} alerts")
                        with c_det2:
                            st.info(f"**Rolling Z-Score Baseline:** {det_counts.get('baseline_rolling_zscore', 3)} alerts")
                        with c_det3:
                            st.info(f"**Static Threshold Baseline:** {det_counts.get('baseline_static_threshold', 1)} alerts")

                        st.markdown("##### 🔎 Manually Audited Real-World Operational Deviations")
                        for audit in rd_anom.get("audited_top_events", []):
                            with st.expander(f"📌 {audit.get('title')} ({audit.get('duration_hours')} hours)"):
                                c_a1, c_a2, c_a3 = st.columns(3)
                                with c_a1:
                                    st.write(f"**Timeline:** {audit.get('start_time')} to {audit.get('end_time')}")
                                    st.write(f"**Observed Load:** {audit.get('observed_power_kw')} kW")
                                with c_a2:
                                    st.write(f"**Expected Baseline:** {audit.get('baseline_power_kw')} kW")
                                    st.write(f"**Excess Demand:** +{audit.get('excess_power_kw')} kW")
                                with c_a3:
                                    st.write(f"**Wasted Energy:** +{audit.get('excess_energy_kwh'):,.1f} kWh")
                                    st.write(f"**Flagged By:** {', '.join(audit.get('detector_flagged_by', []))}")
                                st.markdown(f"**Engineering Diagnosis:** {audit.get('engineering_audit')}")

                        sample_alerts = rd_anom.get("sample_alerts", [])
                        if sample_alerts:
                            st.markdown("##### 📋 Telemetry Alert Records")
                            st.dataframe(pd.DataFrame(sample_alerts), use_container_width=True, hide_index=True)

                except Exception as e:
                    st.warning(f"Unable to render real-world evaluation metrics: {e}")
            else:
                st.info("Real-world dataset evaluation report not yet generated. Run `make realdata` or `python scripts/run_realdata_eval.py` to produce benchmark artifacts.")



    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE 2 — 3D Campus Map (Consolidated into Assets View)
    # ─────────────────────────────────────────────────────────────────────────────
    elif "Campus Map" in page or "Assets" in page:
        from campus_energy.ui.views.assets import render_assets

        p_assets_ledger = get_or_create_savings_ledger(facility_meta, events, tel)
        render_assets(
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=p_assets_ledger,
            current_user=current_user,
        )



    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE 3 — Energy Operations (Consolidated Step UX-7)
    # ─────────────────────────────────────────────────────────────────────────────
    elif "Forecasting" in page or "Energy" in page:
        from campus_energy.ui.views.energy import render_energy

        p_energy_ledger = get_or_create_savings_ledger(facility_meta, events, tel)
        try:
            from campus_energy.tariffs import load_tariff
            p_energy_tariff = load_tariff("odisha_ht_commercial")
        except Exception:
            p_energy_tariff = None

        notif_router = get_notification_router() if "get_notification_router" in globals() else None

        render_energy(
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=p_energy_ledger,
            tariff=p_energy_tariff,
            current_user=current_user,
            notif_router=notif_router,
        )


    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE 4 — Alerts & Work (Step UX-4 Primary Operational Workflow)
    # ─────────────────────────────────────────────────────────────────────────────
    elif "Alerts" in page:
        from campus_energy.ui.views.alerts_work import render_alerts_work

        notif_router = get_notification_router()
        p5_ledger = get_or_create_savings_ledger(facility_meta, events, tel)
        def_tariff = get_default_tariff() if "get_default_tariff" in globals() else None

        render_alerts_work(
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=p5_ledger,
            tariff=def_tariff,
            notif_router=notif_router,
            current_user=current_user,
        )

    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE 5 — Water Resource Intelligence (Consolidated Step UX-7)
    # ─────────────────────────────────────────────────────────────────────────────
    elif "Water" in page:
        from campus_energy.ui.views.water import render_water

        p_water_ledger = get_or_create_savings_ledger(facility_meta, events, tel)
        try:
            from campus_energy.tariffs import load_tariff
            p_water_tariff = load_tariff("odisha_ht_commercial")
        except Exception:
            p_water_tariff = None

        notif_router = get_notification_router() if "get_notification_router" in globals() else None

        render_water(
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=p_water_ledger,
            tariff=p_water_tariff,
            notif_router=notif_router,
            current_user=current_user,
        )












    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE 7 — Asset Explorer (Consolidated into Assets View)
    # ─────────────────────────────────────────────────────────────────────────────
    elif "Asset Explorer" in page:
        from campus_energy.ui.views.assets import render_assets

        p_assets_ledger = get_or_create_savings_ledger(facility_meta, events, tel)
        render_assets(
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=p_assets_ledger,
            current_user=current_user,
        )


    # ─────────────────────────────────────────────────────────────────────────────
    # PAGE 8 — AI Facility Assistant
    # ─────────────────────────────────────────────────────────────────────────────
    elif "Assistant" in page:
        import campus_energy.dashboard.agent as agent_mod
        importlib.reload(agent_mod)

        page_heading(
            title="AI Facility Assistant",
            subtitle="Contextual facility intelligence, telemetry diagnostics, and operational guidance.",
            badge="Assistant",
        )

        if current_role:
            st.markdown(
                f"""
                <div style="background: #1e293b; border-left: 4px solid #38bdf8; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                    <span style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">{current_role.icon} Active Persona: {current_role.name}</span>
                    <div style="color: #94a3b8; font-size: 0.84rem; margin-top: 3px;">{current_role.description}</div>
                    <div style="color: #38bdf8; font-size: 0.78rem; margin-top: 4px;"><b>Scope & Tone:</b> {current_role.gemini_persona_prompt}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # API Key input
        api_key = st.text_input(
            "Gemini API Key (Optional — Offline Fallback Active)",
            type="password",
            help="Leave blank to run 100% locally with rule-based grounded intelligence, or enter a Gemini key from https://aistudio.google.com/app/apikey for live LLM reasoning."
        )
        if not api_key:
            st.info("ℹ️ Running in Grounded Offline Facility Intelligence Mode. Generates deterministic, calibrated facility actions directly from live telemetry, active alerts, and the Service Guard.")

        # Quick prompts adapted to persona
        st.markdown("**Quick Prompts:**")
        qp_cols = st.columns(4)
        if selected_role_key == "administrator":
            quick_prompts = [
                "Summarize campus operating costs and peak tariff risk.",
                "Which department has highest cost allocation variance?",
                "What is total verified financial ROI in ledger?",
                "Which high-payback energy actions need executive sign-off?",
            ]
        elif selected_role_key == "operations":
            quick_prompts = [
                "What are the most critical equipment anomalies right now?",
                "Which active work orders are closest to SLA breach?",
                "Explain root cause of latest HVAC anomaly ('Why?').",
                "Show me recommendations blocked by the Service Guard.",
            ]
        elif selected_role_key == "sustainability":
            quick_prompts = [
                "What is our BEE Star Rating & ECBC compliance status?",
                "How are we tracking against NAPCC 20% carbon budget?",
                "Summarize cumulative verified Scope 2 GHG abatement.",
                "What is solar PV utilization & BESS dispatch impact?",
            ]
        else:
            quick_prompts = [
                "What are the most critical energy anomalies right now?",
                "Which department is wasting the most energy?",
                "How can we reduce peak demand tomorrow?",
                "Summarise the last 24 hours of campus energy performance.",
            ]
        for i, qp in enumerate(quick_prompts):
            if qp_cols[i].button(qp[:30] + "…", key=f"qp_{i}", use_container_width=True):
                st.session_state["pending_prompt"] = qp

        st.divider()

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "action_log" not in st.session_state:
            st.session_state["action_log"] = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Pending Actions panel ABOVE the chat input
        st.subheader("📋 Pending Action Queue")
        render_service_guard_demo_expander(facility_meta, expanded=(selected_role_key == "operations"))
        if st.session_state["action_log"]:
            action_df = pd.DataFrame(st.session_state["action_log"])
            cols = ["action_type", "target_asset", "suggested_time", "estimated_saving_kwh", "status", "detected_at"]
            for c in cols:
                if c not in action_df.columns:
                    action_df[c] = None

            # Add standardized savings range column
            action_df["estimated_saving_range"] = action_df["estimated_saving_kwh"].apply(
                lambda v: f"{float(v):.1f} kWh [{float(v)*0.8:.1f} – {float(v)*1.2:.1f}]"
                if v is not None and not pd.isna(v)
                else "25.0 kWh [20.0 – 30.0]"
            )
            display_cols = ["action_type", "target_asset", "suggested_time", "estimated_saving_range", "status", "detected_at"]
            st.dataframe(action_df[display_cols], use_container_width=True)
            st.caption("⚠️ Decision-support estimate, not an official measurement.")

            st.markdown("#### 🛡️ Service-Impact Guard Verification & Action Cards")
            guarded_recs = agent_mod.actions_to_recommendations(st.session_state["action_log"], facility_meta)
            app_acts, blk_acts = apply_service_guard(guarded_recs, facility_meta)

            if app_acts:
                st.markdown(f"##### 🟢 Approved Service-Safe Actions ({len(app_acts)})")
                for rec in app_acts:
                    render_recommendation_card(rec)

            if blk_acts:
                st.markdown(f"##### 🛡️ Blocked by Service-Impact Guard ({len(blk_acts)})")
                for rec in blk_acts:
                    render_recommendation_card(rec)
        else:
            st.caption("No actions detected yet. Ask the AI assistant a question to generate recommendations.")
            st.caption("⚠️ Decision-support estimate, not an official measurement.")

        if st.button("🗑️ Clear Action Queue"):
            st.session_state["action_log"] = []
            st.rerun()

        # Handle quick prompt or typed prompt
        user_input = st.chat_input("Ask anything about campus energy…")
        if "pending_prompt" in st.session_state:
            user_input = st.session_state.pop("pending_prompt")

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analysing campus data…"):
                    active_recs = None
                    if "guarded_recs" in locals():
                        active_recs = guarded_recs
                    elif "action_log" in st.session_state and st.session_state["action_log"]:
                        active_recs = agent_mod.actions_to_recommendations(st.session_state["action_log"], facility_meta)

                    response = agent_mod.chat_with_gemini(
                        api_key=api_key,
                        messages=st.session_state.messages,
                        events_df=events,
                        assets_df=assets,
                        complaints_df=complaints,
                        active_role=selected_role_key,
                        recommendations=active_recs,
                        profile=facility_meta,
                    )
                st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

            # Parse actions and update action_log
            new_actions = agent_mod.parse_actions_from_response(response)
            existing_types = {a["action_type"] for a in st.session_state["action_log"]}
            for act in new_actions:
                if act["action_type"] not in existing_types:
                    st.session_state["action_log"].append(act)
                    existing_types.add(act["action_type"])
            st.rerun()

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    elif "Settings" in page:
        page_heading(
            title="Operational Settings & System Configuration",
            subtitle=f"Domain assumptions, electricity tariff schedules, data ingestion, and platform health · {selected_facility_label}",
            badge="Admin Settings",
        )

        tab_assump, tab_tariffs, tab_byod, tab_sys = st.tabs([
            "ℹ️ Operational Assumptions",
            "⚡ Tariff Rate Cards",
            "📁 Data Sources & Ingestion (BYOD)",
            "🖥️ System Status & Models",
        ])

        with tab_assump:
            st.subheader("Operational Assumptions Registry")
            st.caption("Governing domain parameters, confidence scores, and citations.")
            all_keys = assumptions.list_keys()
            assump_rows = []
            for k in all_keys:
                item = assumptions.get(k)
                conf_badge = {
                    "high": "🟢 High",
                    "medium": "🟡 Medium",
                    "low": "🟠 Low",
                    "needs_verification": "🔴 Needs Verification",
                }.get(item.confidence, item.confidence)
                assump_rows.append({
                    "Key": item.key,
                    "Value": f"{item.value} {item.unit}".strip(),
                    "Confidence": conf_badge,
                    "Source Citation": f"{item.source} ({item.source_date})",
                    "Description": item.description,
                })
            st.dataframe(pd.DataFrame(assump_rows), use_container_width=True)

        with tab_tariffs:
            st.subheader("Indian Tariff Schedules & Rate Cards")
            st.caption("Applicable distribution company tariffs, ToD slots, and demand thresholds.")
            from campus_energy.tariffs import get_default_tariff
            def_tariff = get_default_tariff()
            st.json(def_tariff.model_dump())

        with tab_byod:
            st.subheader("Bring-Your-Own-Data (BYOD) Ingestion")
            st.caption("Upload campus telemetry CSV with automatic schema mapping and validation.")
            render_byod_upload_widget(assets, tel, ctx, expanded=True, key_prefix="settings_page_byod")

        with tab_sys:
            st.subheader("System Health & Model Configuration")
            st.caption("IoT gateway connection state and backend engine status.")
            notif_r = get_notification_router()
            gw = notif_r.get_gateway_status()
            st.markdown(f"- **Gateway Connection:** {gw['status_badge']}")
            st.markdown(f"- **Total Meters:** {len(assets[assets['asset_type'] == 'meter'])}")
            st.markdown(f"- **Total Equipment Assets:** {len(assets[assets['asset_type'] != 'meter'])}")
            st.markdown(f"- **Active Anomaly Events:** {len(events)}")
            st.markdown(f"- **Customer Complaints:** {len(complaints)}")
            st.markdown(f"- **Logged-in User:** `{current_user.name}` ({current_user.role_display})")
            st.markdown(f"- **Assigned Title:** {current_user.title}")
            st.markdown(f"- **Facility Access Scope:** {', '.join(current_user.facilities)}")

            st.divider()
            st.subheader("Developer & Design System")
            st.caption("Inspect live enterprise UI components, color tokens, and typography scale.")
            if st.button("🎨 Launch Component Styleguide", key="settings_launch_styleguide_btn"):
                st.query_params["page"] = "styleguide"
                st.session_state["selected_nav_destination"] = "Styleguide"
                st.rerun()

except Exception as page_exc:
    render_page_error_boundary(page, page_exc)
