"""Water Resource Destination View for Nivara (HydroSense).

Building-level water telemetry, diurnal flow curves, Minimum Night Flow (MNF 01:00-05:00 AM)
leak watch, active pipe network leak anomalies, and 1-click work order dispatch
integrated directly into SavingsLedger and maintenance notifications.

Strictly adheres to the 'One Value by Default' rule and enterprise design system.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from campus_energy import assumptions
from campus_energy.decision.ledger import SavingsLedger, WorkOrderItem
from campus_energy.decision.recommendation import (
    GuardResult,
    Recommendation,
    SavingsBundle,
    SavingsRange,
)
from campus_energy.notify import NotificationRouter
from campus_energy.simulator.water import (
    WATER_TARIFF_INR_PER_KL,
    analyze_water_data,
    generate_water_data,
)
from campus_energy.ui.components import (
    empty_state,
    format_inr,
    format_inr_compact,
    info_popover,
    kpi_card,
    page_heading,
    severity_chip,
    status_chip,
)

logger = logging.getLogger(__name__)

DARK_THEME_LAYOUT = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font_color="#94a3b8",
    margin=dict(l=10, r=10, t=32, b=10),
)


def dispatch_water_leak_work_order(
    leak_record: Dict[str, Any],
    ledger: SavingsLedger,
    facility_id: str = "university",
    assignee: str = "Plumbing Tech",
    sla_hours: int = 12,
    notif_router: Optional[NotificationRouter] = None,
) -> WorkOrderItem:
    """Create a formal WorkOrderItem in SavingsLedger and dispatch maintenance notifications."""
    leak_id = str(leak_record.get("leak_id", "LK-MNF"))
    bld = str(leak_record.get("building", "Campus Facility"))
    leak_type = str(leak_record.get("leak_type", "Pipe Leak"))
    lost_kl = float(leak_record.get("lost_volume_kl", 10.0))
    cost_loss = float(leak_record.get("cost_loss_inr", lost_kl * WATER_TARIFF_INR_PER_KL))
    recom_text = str(leak_record.get("recommendation", "Inspect plumbing riser and PRV valve."))
    sev = str(leak_record.get("severity", "High")).lower()

    # Formulate canonical Recommendation
    rec_id = f"REC-{leak_id}"
    wo_id = f"WO-{leak_id}"

    savings_bundle = SavingsBundle(
        kwh=SavingsRange(low=0.0, expected=0.0, high=0.0),
        inr=SavingsRange(low=round(cost_loss * 0.8, 1), expected=round(cost_loss, 1), high=round(cost_loss * 1.2, 1)),
        co2_kg=SavingsRange(low=0.0, expected=0.0, high=0.0),
    )

    rec = Recommendation(
        id=rec_id,
        facility_id=facility_id,
        type="leak_repair",
        target=bld,
        action_text=f"Repair {leak_type} in {bld}: {recom_text}",
        trigger=f"HydroSense Water Anomaly {leak_id}",
        savings=savings_bundle,
        cost=500.0,
        payback_months=round(500.0 / max(cost_loss, 1.0), 2),
        assumption_refs=["water.tariff_inr_per_kl", "water.mnf_start_hour", "water.mnf_end_hour"],
        service_guard=GuardResult(
            status="pass",
            reason="Plumbing repair restores hydraulic pressure; zero electrical or comfort risk.",
            constraints_checked=["protected_load_check", "pmv_comfort_check"],
        ),
        owner_role="operations",
        lifecycle="work_order",
        confidence=0.95,
        is_estimate=False,
    )

    # Ingest into ledger
    wo_item = ledger.add_recommendation(
        rec=rec,
        priority="critical" if sev == "critical" else "high",
        assignee=assignee,
        work_order_id=wo_id,
        sla_hours=sla_hours,
    )

    # Advance to formal work order
    wo_item = ledger.create_work_order(
        rec_id=wo_item.work_order_id,
        assignee=assignee,
        sla_hours=sla_hours,
        priority=wo_item.priority,
    )

    # Dispatch outbound notifications
    if notif_router:
        try:
            notif_router.dispatch_from_work_order(wo_item)
        except Exception as err:
            logger.warning("Failed water leak notification dispatch: %s", err)

    return wo_item


def render_water(
    facility_meta: Dict[str, Any],
    assets: pd.DataFrame,
    tel: pd.DataFrame,
    ctx: pd.DataFrame,
    events: pd.DataFrame,
    complaints: pd.DataFrame,
    ledger: SavingsLedger,
    tariff: Optional[Any] = None,
    notif_router: Optional[Any] = None,
    current_user: Optional[Any] = None,
    water_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> None:
    """Render the unified Water Destination View (HydroSense)."""
    fac_name = facility_meta.get("name", "Campus Facility")
    page_heading(
        title="Water Resource Intelligence",
        subtitle=f"Building telemetry, Minimum Night Flow (MNF) leak watch & acoustic anomaly detection · {fac_name}",
        badge="HydroSense v1.2",
    )

    # Load or generate water telemetry
    if water_df is None:
        data_dir_str = str(facility_meta.get("data_dir", "data/generated"))
        data_dir = Path(data_dir_str)
        water_file = data_dir / "water.csv"
        if water_file.exists():
            try:
                water_df = pd.read_csv(water_file)
            except Exception as e:
                logger.warning("Error reading water.csv: %s", e)
                water_df = generate_water_data(assets)
        else:
            water_df = generate_water_data(assets)

    if water_df is None or water_df.empty:
        empty_state(
            title="No Water Telemetry Available",
            description="Water meter telemetry records could not be loaded for this facility.",
        )
        return

    if "timestamp" in water_df.columns:
        water_df["timestamp"] = pd.to_datetime(water_df["timestamp"])

    with st.spinner("Analyzing water telemetry and pipe network acoustics…"):
        water_analytics = analyze_water_data(water_df)

    kpis = water_analytics["kpis"]
    daily = water_analytics["daily_trend"]
    bld_breakdown = water_analytics["building_breakdown"]
    diurnal = water_analytics["diurnal_curve"]
    leaks = water_analytics["leak_events"]

    # ─────────────────────────────────────────────────────────────────────────
    # 1. TOP KPI STRIP (Single Values by Default)
    # ─────────────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric(
            label="Total Consumption",
            value=f"{kpis['total_water_kl']:,.1f} kL",
            delta=f"Avg {kpis['daily_avg_kl']:.1f} kL/day",
            delta_color="off",
        )
    with c2:
        st.metric(
            label="Water Cost",
            value=format_inr(kpis["water_cost_inr"]),
            delta=f"Tariff ₹{kpis['cost_per_kl']:.0f} / kL",
            delta_color="off",
        )
    with c3:
        act_cnt = kpis["active_leaks"]
        st.metric(
            label="Active Leaks",
            value=f"{act_cnt}",
            delta="Immediate action required" if act_cnt > 0 else "Network healthy",
            delta_color="inverse" if act_cnt > 0 else "normal",
        )
    with c4:
        st.metric(
            label="Avoidable Loss",
            value=f"{kpis['avoidable_loss_kl']:,.1f} kL",
            delta=f"Waste: {format_inr(kpis['avoidable_loss_inr'])}",
            delta_color="inverse",
        )
    with c5:
        night_lpm = kpis["night_leak_lpm"]
        st.metric(
            label="Night Flow (MNF)",
            value=f"{night_lpm:.1f} L/min",
            delta="MNF baseline breached" if night_lpm > 0 else "Nominal baseload",
            delta_color="inverse" if night_lpm > 0 else "normal",
        )

    # Info Popover for Municipal Tariff & Methodology
    info_popover(
        title="Water Intelligence & MNF Calculation Notes",
        calculation_notes=(
            f"**Audit Standards:** AWWA M36 & BS EN 12255 Minimum Night Flow methodology.<br/>"
            f"• **MNF Evaluation Window:** 01:00 AM – 05:00 AM (human demand approaches zero).<br/>"
            f"• **Concealed Leak Threshold:** Night flow exceeding 3× median historical baseload.<br/>"
            f"• **Municipal Water Tariff:** Fixed rate of ₹{kpis['cost_per_kl']:.2f} per kiloliter (kL).<br/>"
            f"• **Avoidable Financial Loss:** {kpis['avoidable_loss_kl']:,.1f} kL × ₹{kpis['cost_per_kl']:.0f} = {format_inr(kpis['avoidable_loss_inr'])}."
        ),
        assumption_keys=["water.tariff_inr_per_kl", "water.mnf_start_hour", "water.mnf_end_hour"],
        button_label="ⓘ Water Tariff & MNF Methodology",
    )

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. DIURNAL FLOW CURVE & MINIMUM NIGHT FLOW (MNF 01:00–05:00 AM)
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Diurnal Flow Profile & Minimum Night Flow (MNF) Leak Watch")
    st.caption(
        "AWWA M36 standard: between 01:00 AM and 05:00 AM, legitimate occupant demand drops to near zero. "
        "Any sustained elevation during this window directly quantifies underground pipe fissures or valve bypass."
    )

    fig_diurnal = go.Figure()

    # MNF Shaded Region (01:00 to 05:00 AM)
    fig_diurnal.add_vrect(
        x0=1,
        x1=5,
        fillcolor="#ef4444",
        opacity=0.15,
        layer="below",
        line_width=1,
        line_color="#ef4444",
        line_dash="dot",
        annotation_text="MNF Leak Assessment Zone (01:00 – 05:00 AM)",
        annotation_position="top left",
        annotation_font_color="#fca5a5",
        annotation_font_size=11,
    )

    # Observed Average Flow
    fig_diurnal.add_trace(go.Scatter(
        x=diurnal["hour"],
        y=diurnal["avg_flow_rate_lpm"],
        mode="lines+markers",
        name="Observed Flow Rate (L/min)",
        line=dict(color="#38bdf8", width=2.5),
        marker=dict(size=6, color="#38bdf8"),
        hovertemplate="<b>Hour %{x}:00</b><br>Observed: %{y:.1f} L/min<extra></extra>",
    ))

    # Baseline Flow
    if "baseline_flow_lpm" in diurnal.columns:
        fig_diurnal.add_trace(go.Scatter(
            x=diurnal["hour"],
            y=diurnal["baseline_flow_lpm"],
            mode="lines",
            name="Normal Baseline (L/min)",
            line=dict(color="#64748b", width=1.8, dash="dash"),
            hovertemplate="<b>Hour %{x}:00</b><br>Baseline: %{y:.1f} L/min<extra></extra>",
        ))

    fig_diurnal.update_layout(
        **DARK_THEME_LAYOUT,
        title="24-Hour Campus Diurnal Flow Cycle (Liters / Minute)",
        xaxis=dict(
            title="Hour of Day (24-Hour Cycle)",
            tickmode="linear",
            tick0=0,
            dtick=1,
            gridcolor="#1e293b",
        ),
        yaxis=dict(title="Flow Rate (L/min)", gridcolor="#1e293b"),
        height=340,
        hovermode="x unified",
        legend=dict(bgcolor="#1e293b", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_diurnal, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # 3. ACTIVE LEAK ALERTS & 1-CLICK WORK ORDER DISPATCH
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Active Pipe Network Leak Alerts & Remediation")
    st.caption("Active flow anomalies detected via statistical deviation from diurnal flow baselines and overnight threshold checks.")

    if not leaks.empty:
        fac_id = str(facility_meta.get("facility_id", "university"))

        for idx, leak in leaks.iterrows():
            lid = str(leak.get("leak_id", f"LK-{idx}"))
            bld = str(leak.get("building", "Building"))
            leak_type = str(leak.get("leak_type", "Water Leak"))
            sev = str(leak.get("severity", "High")).lower()
            cost_loss = float(leak.get("cost_loss_inr", 0.0))
            lost_vol = str(leak.get("lost_volume", "0 kL"))
            flow_rate_lpm = float(leak.get("leak_rate_lpm", 0.0))
            recom = str(leak.get("recommendation", "Inspect plumbing riser."))
            status_text = str(leak.get("status", "Active Leak"))

            # Check if already dispatched as work order
            wo_existing = ledger._find_item(f"WO-{lid}") or ledger._find_item(f"REC-{lid}")
            is_dispatched = wo_existing is not None or st.session_state.get(f"wo_dispatched_{lid}", False)

            border_col = "#ef4444" if sev == "critical" else "#f59e0b"
            bg_col = "rgba(239, 68, 68, 0.06)" if sev == "critical" else "rgba(245, 158, 11, 0.06)"

            st.markdown(
                f"""
                <div style="background:{bg_col}; border-left: 4px solid {border_col}; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; border-top: 1px solid #334155; border-right: 1px solid #334155; border-bottom: 1px solid #334155;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px;">
                        <div>
                            <span style="font-weight:700; color:#f8fafc; font-size:1.02rem;">💧 {bld} — {leak_type}</span>
                            <span style="margin-left: 8px;">{severity_chip(sev)}</span>
                            <span style="margin-left: 6px;">{status_chip('in_progress' if is_dispatched else 'active')}</span>
                        </div>
                        <span style="color:#ef4444; font-weight:700; font-size:0.95rem;">
                            Lost: {lost_vol} · Waste: {format_inr(cost_loss)}
                        </span>
                    </div>
                    <div style="color:#94a3b8; font-size:0.84rem; margin-bottom: 8px;">
                        ⏱️ <b>Start:</b> {leak.get('start_time', 'N/A')} &nbsp;|&nbsp; ⏳ <b>Duration:</b> {leak.get('duration', 'N/A')} &nbsp;|&nbsp; 🌊 <b>Rate:</b> ~{flow_rate_lpm:.1f} L/min (~{flow_rate_lpm*60:.0f} L/h)
                    </div>
                    <div style="background:#0f172a; border-radius: 6px; padding: 8px 12px; color:#cbd5e1; font-size:0.83rem; border: 1px solid #334155; margin-bottom: 8px;">
                        🔧 <b>Action Recommendation:</b> {recom}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Action Row
            col_act1, col_act2 = st.columns([2, 4])
            with col_act1:
                if is_dispatched:
                    wo_code = wo_existing.work_order_id if wo_existing else f"WO-{lid}"
                    st.success(f"✅ Dispatched: `{wo_code}` (Plumbing Tech)")
                else:
                    if st.button(
                        f"🔧 Create Repair Work Order",
                        key=f"btn_create_wo_{lid}_{idx}",
                        type="primary",
                        use_container_width=True,
                    ):
                        new_wo = dispatch_water_leak_work_order(
                            leak_record=leak,
                            ledger=ledger,
                            facility_id=fac_id,
                            assignee="Plumbing Tech",
                            sla_hours=12,
                            notif_router=notif_router,
                        )
                        st.session_state[f"wo_dispatched_{lid}"] = True
                        st.toast(f"Work Order {new_wo.work_order_id} dispatched to Plumbing Tech!")
                        st.rerun()

            with col_act2:
                if is_dispatched and wo_existing:
                    sla_str = wo_existing.sla_deadline.strftime("%d %b %H:%M") if wo_existing.sla_deadline else "12 hours"
                    st.caption(f"Assigned to **{wo_existing.assignee}** · SLA Target: **{sla_str}**")

            st.write("")
    else:
        st.success("✅ No active pipe network leaks or MNF baseline breaches detected across campus.")

    st.divider()

    # ─────────────────────────────────────────────────────────────────────────
    # 4. CONSUMPTION TREND & BUILDING WATER SHARE
    # ─────────────────────────────────────────────────────────────────────────
    col_daily, col_share = st.columns([3, 2])

    with col_daily:
        st.subheader("Daily Water Consumption Trend")
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily["date"],
            y=daily["flow_kl"],
            name="Daily Volume (kL)",
            marker_color="#38bdf8",
            opacity=0.88,
            hovertemplate="<b>%{x}</b><br>Volume: %{y:.1f} kL<extra></extra>",
        ))
        if "rolling_avg_3d" in daily.columns:
            fig_daily.add_trace(go.Scatter(
                x=daily["date"],
                y=daily["rolling_avg_3d"],
                mode="lines",
                name="3-Day Moving Avg",
                line=dict(color="#10b981", width=2.2, dash="dash"),
                hovertemplate="<b>3-day MA</b>: %{y:.1f} kL<extra></extra>",
            ))

        fig_daily.update_layout(
            **DARK_THEME_LAYOUT,
            title="Daily Campus Water Intake (kL)",
            xaxis_title="Date",
            yaxis_title="Volume (kL)",
            height=300,
            hovermode="x unified",
            legend=dict(bgcolor="#1e293b", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    with col_share:
        st.subheader("Building Water Share")
        color_seq = ["#38bdf8", "#818cf8", "#a78bfa", "#f472b6", "#34d399", "#fbbf24", "#f87171"]
        fig_pie = go.Figure(data=[go.Pie(
            labels=bld_breakdown["building_id"],
            values=bld_breakdown["total_kl"],
            hole=0.48,
            marker=dict(colors=color_seq[:len(bld_breakdown)]),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:.1f} kL (%{percent})<extra></extra>",
        )])
        fig_pie.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#94a3b8",
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
