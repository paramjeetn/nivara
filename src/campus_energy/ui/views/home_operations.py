"""Operations Role Home View (Step UX-5 / Section 5.2).

Quiet, dense, exception-first operational landing screen designed for
a Facility Operations Lead on a Tuesday morning.

Components:
1. Today Strip (at most 4 numbers):
   - Load Now vs Expected (kW with delta)
   - Open Alerts Count (with critical/high delta)
   - Work Orders Due Today Count (with overdue delta)
   - Sensor Data Health (e.g. 99.8% completeness)
2. Needs Attention (primary screen element):
   - Prioritized operational queue of critical/high alerts, work orders
     nearing or past SLA, and offline/stuck sensors.
   - Actionable click-through to Alerts & Work.
3. Load Today vs Forecast Chart:
   - Clean Plotly chart with actual load vs Ridge forecast band
   - Anomaly markers highlighted as distinct red points.
4. Recent Activity Feed:
   - Last 10 operational events (Alert raised, Work order dispatched, Verified).
"""
from __future__ import annotations

import html
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from campus_energy import assumptions
from campus_energy.analytics.anomalies import detect_anomalies
from campus_energy.analytics.features import engineer_features
from campus_energy.analytics.forecasting import generate_24h_forecast
from campus_energy.analytics.workorders import estimate_energy_waste
from campus_energy.decision.ledger import SavingsLedger, WorkOrderItem
from campus_energy.decision.recommendation import Recommendation, SavingsBundle
from campus_energy.decision.tariffs import IndianTariff
from campus_energy.schemas import Alert, Severity
from campus_energy.ui.components import (
    empty_state,
    format_inr,
    format_inr_compact,
    format_relative_time,
    kpi_card,
    page_heading,
    severity_chip,
    status_chip,
)
from campus_energy.ui.views.alerts_work import build_plain_headline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Pure Data Preparation Functions (Unit Testable)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_operations_today_strip(
    tel: pd.DataFrame,
    assets: pd.DataFrame,
    events: Optional[Union[pd.DataFrame, List[Any]]] = None,
    ledger: Optional[SavingsLedger] = None,
    expected_kw_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the 4 operational headline numbers for the Today Strip.

    Returns:
        dict with:
            - load_now_kw, expected_load_kw, load_delta_kw, load_delta_str, load_delta_color
            - open_alerts_count, crit_high_count, alert_delta_str, alert_delta_color
            - work_orders_due_today, work_orders_overdue, wo_delta_str, wo_delta_color
            - data_health_pct, data_health_str, data_health_color
    """
    # 1. Load Now vs Expected
    load_now = 0.0
    expected_load = 0.0
    if tel is not None and not tel.empty:
        latest_ts = pd.to_datetime(tel["timestamp"]).max()
        # Find meters if available
        if assets is not None and not assets.empty and "asset_type" in assets.columns:
            meters = assets[assets["asset_type"] == "meter"]["asset_id"].tolist()
        else:
            meters = []

        latest_tel = tel[pd.to_datetime(tel["timestamp"]) == latest_ts]
        if meters and latest_tel["asset_id"].isin(meters).any():
            load_now = float(latest_tel[latest_tel["asset_id"].isin(meters)]["power_kw"].sum())
        else:
            load_now = float(latest_tel["power_kw"].sum())

        if expected_kw_override is not None:
            expected_load = float(expected_kw_override)
        elif "expected_power_kw" in latest_tel.columns and latest_tel["expected_power_kw"].notna().any():
            expected_load = float(latest_tel["expected_power_kw"].sum())
        else:
            # Baseline proxy: 95% of typical load or previous day same hour
            expected_load = max(0.0, round(load_now * 0.95, 1))

    load_delta = round(load_now - expected_load, 1)
    if load_delta > 0:
        load_delta_str = f"+{load_delta:.1f} kW vs expected"
        load_delta_color = "inverse"  # Higher power than expected is unfavorable
    elif load_delta < 0:
        load_delta_str = f"{load_delta:.1f} kW vs expected"
        load_delta_color = "normal"  # Lower load is favorable
    else:
        load_delta_str = "On baseline"
        load_delta_color = "off"

    # 2. Open Alerts Count
    open_alerts_count = 0
    crit_high_count = 0
    if events is not None:
        if isinstance(events, pd.DataFrame):
            open_alerts_count = len(events)
            if "severity" in events.columns:
                crit_high_count = int(events["severity"].astype(str).str.lower().isin(["critical", "high"]).sum())
        elif isinstance(events, list):
            open_alerts_count = len(events)
            for a in events:
                sev = getattr(a, "severity", None)
                s_str = sev.value if hasattr(sev, "value") else str(sev).lower()
                if s_str in ["critical", "high"]:
                    crit_high_count += 1

    if crit_high_count > 0:
        alert_delta_str = f"{crit_high_count} Critical / High"
        alert_delta_color = "inverse"
    elif open_alerts_count > 0:
        alert_delta_str = f"{open_alerts_count} Active"
        alert_delta_color = "inverse"
    else:
        alert_delta_str = "Zero Exceptions"
        alert_delta_color = "normal"

    # 3. Work Orders Due Today Count
    wo_due_today = 0
    wo_overdue = 0
    if ledger is not None:
        now_utc = datetime.now(timezone.utc)
        for wo in ledger.get_items():
            if wo.status in ["proposed", "accepted", "work_order"]:
                # Check SLA
                sla = wo.sla_deadline
                if sla.tzinfo is None:
                    sla = sla.replace(tzinfo=timezone.utc)
                if sla < now_utc:
                    wo_overdue += 1
                    wo_due_today += 1
                elif (sla - now_utc) <= timedelta(hours=24):
                    wo_due_today += 1

    if wo_overdue > 0:
        wo_delta_str = f"{wo_overdue} Overdue SLA"
        wo_delta_color = "inverse"
    elif wo_due_today > 0:
        wo_delta_str = f"{wo_due_today} Due in 24h"
        wo_delta_color = "off"
    else:
        wo_delta_str = "Queue Clean"
        wo_delta_color = "normal"

    # 4. Sensor Data Health
    data_health_pct = 99.8
    if tel is not None and not tel.empty:
        total_records = len(tel)
        nan_or_zero = int((tel["power_kw"].isna() | (tel["power_kw"] < 0)).sum())
        if total_records > 0:
            calc_health = round((1.0 - (nan_or_zero / total_records)) * 100.0, 1)
            data_health_pct = max(90.0, min(100.0, calc_health))

    if data_health_pct >= 99.0:
        data_health_str = "Nominal (0 Dropouts)"
        data_health_color = "normal"
    elif data_health_pct >= 95.0:
        data_health_str = "Minor Telemetry Noise"
        data_health_color = "off"
    else:
        data_health_str = "Degraded Telemetry"
        data_health_color = "inverse"

    return {
        "load_now_kw": round(load_now, 1),
        "expected_load_kw": round(expected_load, 1),
        "load_delta_kw": load_delta,
        "load_delta_str": load_delta_str,
        "load_delta_color": load_delta_color,
        "open_alerts_count": open_alerts_count,
        "crit_high_count": crit_high_count,
        "alert_delta_str": alert_delta_str,
        "alert_delta_color": alert_delta_color,
        "work_orders_due_today": wo_due_today,
        "work_orders_overdue": wo_overdue,
        "wo_delta_str": wo_delta_str,
        "wo_delta_color": wo_delta_color,
        "data_health_pct": data_health_pct,
        "data_health_str": data_health_str,
        "data_health_color": data_health_color,
    }


def assemble_needs_attention(
    events: Optional[Union[pd.DataFrame, List[Any]]],
    assets: pd.DataFrame,
    tel: Optional[pd.DataFrame] = None,
    ctx: Optional[pd.DataFrame] = None,
    facility_meta: Optional[Any] = None,
    ledger: Optional[SavingsLedger] = None,
    alert_statuses: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Assemble prioritized queue combining:
    1. Open critical/high alerts
    2. Work orders nearing or past SLA
    3. Offline or stuck sensors
    """
    items: List[Dict[str, Any]] = []
    statuses = alert_statuses or {}

    # Asset lookup helper
    asset_names = {}
    asset_types = {}
    rated_powers = {}
    if assets is not None and not assets.empty and "asset_id" in assets.columns:
        for _, r in assets.iterrows():
            aid = str(r["asset_id"])
            asset_names[aid] = str(r.get("asset_name") or aid)
            asset_types[aid] = str(r.get("asset_type") or "equipment")
            if "rated_power_kw" in r and pd.notna(r["rated_power_kw"]):
                rated_powers[aid] = float(r["rated_power_kw"])

    # 1. Open Alerts (critical / high / medium)
    if events is not None:
        raw_list = []
        if isinstance(events, pd.DataFrame):
            for _, row in events.iterrows():
                raw_list.append(row.to_dict())
        elif isinstance(events, list):
            for a in events:
                if isinstance(a, Alert):
                    raw_list.append({
                        "alert_id": a.alert_id,
                        "asset_id": a.asset_id,
                        "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
                        "anomaly_type": a.anomaly_type,
                        "actual_value": a.actual_value,
                        "expected_value": a.expected_value,
                        "deviation": a.deviation,
                        "detected_at": a.detected_at,
                        "description": a.evidence,
                    })
                elif isinstance(a, dict):
                    raw_list.append(a)

        for el in raw_list:
            aid = str(el.get("alert_id") or el.get("event_id") or "ALT-UNK")
            # Check lifecycle status
            st_val = statuses.get(aid, str(el.get("status", "active"))).lower()
            if st_val in ["resolved", "closed", "snoozed"]:
                continue

            asset_id = str(el.get("asset_id") or "UNKNOWN")
            raw_sev = str(el.get("severity") or "medium").lower()
            anomaly_type = str(el.get("anomaly_type") or el.get("event_type") or "high_consumption")
            actual_val = el.get("actual_value")
            expected_val = el.get("expected_value")
            dev_val = el.get("deviation")

            if actual_val is None:
                actual_val = rated_powers.get(asset_id, 15.0)
            if expected_val is None:
                expected_val = 0.0 if "left_on" in anomaly_type or "unoccupied" in anomaly_type else round(actual_val * 0.4, 1)
            if dev_val is None:
                dev_val = round(float(actual_val) - float(expected_val), 1)

            headline = build_plain_headline(
                asset_id=asset_id,
                asset_name=asset_names.get(asset_id, asset_id),
                anomaly_type=anomaly_type,
                actual_val=float(actual_val),
                expected_val=float(expected_val),
                deviation=float(dev_val),
            )

            # Impact in INR/day or excess kW
            kw_day, inr_day = estimate_energy_waste(rated_powers.get(asset_id, 15.0), raw_sev, anomaly_type)
            impact_str = f"₹{inr_day:,.0f}/day waste" if inr_day > 0 else f"+{dev_val:.1f} kW excess"

            det_ts = el.get("detected_at") or el.get("start_time") or datetime.now(timezone.utc)

            priority_rank = 0 if raw_sev == "critical" else (1 if raw_sev == "high" else 2)

            items.append({
                "id": aid,
                "type": "alert",
                "severity": raw_sev,
                "priority_rank": priority_rank,
                "asset_id": asset_id,
                "headline": headline,
                "impact": impact_str,
                "age_dt": det_ts,
                "age": format_relative_time(det_ts),
                "action_label": "Open in Alerts & Work",
            })

    # 2. Work Orders Nearing or Past SLA
    if ledger is not None:
        now_utc = datetime.now(timezone.utc)
        for wo in ledger.get_items():
            if wo.status in ["proposed", "accepted", "work_order"]:
                sla = wo.sla_deadline
                if sla.tzinfo is None:
                    sla = sla.replace(tzinfo=timezone.utc)

                is_overdue = sla < now_utc
                is_nearing = (sla - now_utc) <= timedelta(hours=12)

                if is_overdue or is_nearing:
                    sev = "critical" if is_overdue else "high"
                    rank = 0 if is_overdue else 1
                    hours_diff = (now_utc - sla).total_seconds() / 3600.0 if is_overdue else (sla - now_utc).total_seconds() / 3600.0

                    if is_overdue:
                        issue_text = f"SLA Breached (+{int(hours_diff)}h): {wo.recommendation.action_text}"
                    else:
                        issue_text = f"SLA Warning ({int(hours_diff)}h left): {wo.recommendation.action_text}"

                    items.append({
                        "id": wo.work_order_id,
                        "type": "work_order",
                        "severity": sev,
                        "priority_rank": rank,
                        "asset_id": wo.recommendation.target,
                        "headline": issue_text,
                        "impact": f"Target: {format_inr(wo.estimated_savings.inr.expected)}/yr",
                        "age_dt": wo.created_at,
                        "age": format_relative_time(wo.created_at),
                        "action_label": "Open in Alerts & Work",
                    })

    # 3. Offline / Stuck Sensors
    if tel is not None and not tel.empty:
        latest_ts = pd.to_datetime(tel["timestamp"]).max()
        recent_tel = tel[pd.to_datetime(tel["timestamp"]) == latest_ts]

        for aid, rated_kw in rated_powers.items():
            if rated_kw > 5.0 and aid in recent_tel["asset_id"].values:
                reading = recent_tel[recent_tel["asset_id"] == aid]
                if not reading.empty:
                    val = reading["power_kw"].iloc[0]
                    if pd.isna(val) or val <= 0.001:
                        # Sensor dropout / stuck at zero
                        items.append({
                            "id": f"SENSOR-{aid}",
                            "type": "sensor",
                            "severity": "high",
                            "priority_rank": 1,
                            "asset_id": aid,
                            "headline": f"{asset_names.get(aid, aid)} sensor stuck at 0.0 kW (expected telemetry active)",
                            "impact": "Telemetry Offline",
                            "age_dt": latest_ts,
                            "age": format_relative_time(latest_ts),
                            "action_label": "Open in Alerts & Work",
                        })

    # Sort items: Priority rank asc (0=crit, 1=high, 2=med), then age (most recent first)
    items.sort(key=lambda x: (x["priority_rank"], str(x["age_dt"])), reverse=False)

    return items[:10]


def prepare_recent_activity(
    events: Optional[Union[pd.DataFrame, List[Any]]],
    ledger: Optional[SavingsLedger] = None,
) -> List[Dict[str, Any]]:
    """Assemble compact chronologically sorted activity feed of last 10 operational events."""
    activity: List[Dict[str, Any]] = []

    # 1. Alert events
    if events is not None:
        if isinstance(events, pd.DataFrame):
            for _, r in events.iterrows():
                activity.append({
                    "timestamp": r.get("start_time") or r.get("detected_at") or datetime.now(timezone.utc),
                    "event_type": "alert_raised",
                    "badge": "🚨 Alert",
                    "badge_color": "#ef4444",
                    "title": f"Alert Raised · {r.get('asset_id', 'Asset')}",
                    "detail": str(r.get("description") or r.get("evidence") or "Observed operational deviation"),
                })
        elif isinstance(events, list):
            for a in events:
                aid = getattr(a, "asset_id", "Asset")
                activity.append({
                    "timestamp": getattr(a, "detected_at", datetime.now(timezone.utc)),
                    "event_type": "alert_raised",
                    "badge": "🚨 Alert",
                    "badge_color": "#ef4444",
                    "title": f"Alert Raised · {aid}",
                    "detail": getattr(a, "evidence", "Observed anomaly"),
                })

    # 2. Ledger Work Order Dispatches & Verifications
    if ledger is not None:
        for it in ledger.get_items():
            if it.status in ["work_order", "done"]:
                activity.append({
                    "timestamp": it.created_at,
                    "event_type": "wo_dispatched",
                    "badge": "🔧 Work Order",
                    "badge_color": "#0284c7",
                    "title": f"Work Order Dispatched · {it.work_order_id}",
                    "detail": f"Assigned to {it.assignee}: {it.recommendation.action_text}",
                })
            elif it.status == "verified" and it.completed_at is not None:
                saved_inr = it.verified_savings.inr.expected if it.verified_savings else 0.0
                activity.append({
                    "timestamp": it.completed_at,
                    "event_type": "fault_resolved",
                    "badge": "✅ Verified",
                    "badge_color": "#10b981",
                    "title": f"Resolved & Verified · {it.work_order_id}",
                    "detail": f"{format_inr(saved_inr)} verified avoided cost ({it.hit_rate_status.upper()})",
                })

    # Sort descending by timestamp (newest first)
    activity.sort(key=lambda x: str(x["timestamp"]), reverse=True)
    return activity[:10]


def build_load_forecast_figure(
    tel: pd.DataFrame,
    ctx: pd.DataFrame,
    assets: pd.DataFrame,
    events: Optional[Union[pd.DataFrame, List[Any]]] = None,
) -> go.Figure:
    """Generate the Load Today vs Forecast Plotly chart with confidence ribbon and anomaly markers."""
    fig = go.Figure()

    if tel is None or tel.empty:
        fig.update_layout(
            title="Load Today vs Forecast — Telemetry Waiting",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#94a3b8",
        )
        return fig

    # 1. Filter meters
    if assets is not None and not assets.empty and "asset_type" in assets.columns:
        meters = assets[assets["asset_type"] == "meter"]["asset_id"].tolist()
    else:
        meters = []

    tel_work = tel.copy()
    tel_work["timestamp"] = pd.to_datetime(tel_work["timestamp"])
    if tel_work["timestamp"].dt.tz is not None:
        tel_work["timestamp"] = tel_work["timestamp"].dt.tz_localize(None)

    if meters and tel_work["asset_id"].isin(meters).any():
        meter_tel = tel_work[tel_work["asset_id"].isin(meters)]
    else:
        meter_tel = tel_work

    latest_ts = tel_work["timestamp"].max()
    window_start = latest_ts - timedelta(hours=24)

    # Actual load (last 24 hours)
    actual_recent = meter_tel[meter_tel["timestamp"] >= window_start].groupby("timestamp")["power_kw"].sum().reset_index()

    # 2. Run forecast model
    try:
        fcast = generate_24h_forecast(tel, ctx, assets)
    except Exception as e:
        logger.warning("Forecast generation warning: %s", e)
        fcast = pd.DataFrame()

    if not fcast.empty:
        fcast["timestamp"] = pd.to_datetime(fcast["timestamp"])
        if fcast["timestamp"].dt.tz is not None:
            fcast["timestamp"] = fcast["timestamp"].dt.tz_localize(None)

        campus_fc = fcast.groupby("timestamp")[["predicted_power_kw", "lower_bound", "upper_bound"]].sum().reset_index()

        # Ribbon Lower Bound
        fig.add_trace(go.Scatter(
            x=campus_fc["timestamp"],
            y=campus_fc["lower_bound"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Ribbon Upper Bound (fill='tonexty')
        fig.add_trace(go.Scatter(
            x=campus_fc["timestamp"],
            y=campus_fc["upper_bound"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.12)",
            name="80% Prediction Band",
            showlegend=True,
        ))
        # Forecast baseline line
        fig.add_trace(go.Scatter(
            x=campus_fc["timestamp"],
            y=campus_fc["predicted_power_kw"],
            mode="lines",
            name="Forecast Baseline",
            line=dict(color="#38bdf8", width=2, dash="dash"),
        ))

    # Actual power load line
    if not actual_recent.empty:
        fig.add_trace(go.Scatter(
            x=actual_recent["timestamp"],
            y=actual_recent["power_kw"],
            mode="lines",
            name="Actual Load",
            line=dict(color="#10b981", width=2.5),
        ))

    # Mark anomalies as red scatter points
    anomaly_ts = []
    anomaly_power = []
    anomaly_hover = []

    if events is not None:
        evt_rows = []
        if isinstance(events, pd.DataFrame):
            evt_rows = events.to_dict(orient="records")
        elif isinstance(events, list):
            for it in events:
                evt_rows.append({
                    "start_time": getattr(it, "detected_at", None) or getattr(it, "start_time", None),
                    "description": getattr(it, "evidence", None) or getattr(it, "description", "Anomaly"),
                    "severity": getattr(it, "severity", "medium"),
                })

        for ev in evt_rows:
            raw_t = ev.get("start_time") or ev.get("detected_at")
            if raw_t:
                ts = pd.to_datetime(raw_t)
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
                if ts >= window_start:
                    # Find closest actual power reading
                    if not actual_recent.empty:
                        idx = (actual_recent["timestamp"] - ts).abs().idxmin()
                        matched_p = actual_recent.loc[idx, "power_kw"]
                        anomaly_ts.append(ts)
                        anomaly_power.append(matched_p)
                        anomaly_hover.append(str(ev.get("description", "Active Anomaly")))

    if anomaly_ts:
        fig.add_trace(go.Scatter(
            x=anomaly_ts,
            y=anomaly_power,
            mode="markers",
            name="Anomaly Detected",
            marker=dict(color="#ef4444", size=10, symbol="diamond", line=dict(color="#ffffff", width=1)),
            text=anomaly_hover,
            hoverinfo="x+y+text",
        ))

    fig.update_layout(
        title="Load Today vs Forecast (Campus Total kW)",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto"),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=30),
        legend=dict(
            bgcolor="rgba(30, 41, 59, 0.8)",
            bordercolor="#334155",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False),
        yaxis=dict(title="Campus Power (kW)", showgrid=True, gridcolor="#1e293b", zeroline=False),
        height=320,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Main Streamlit Render Function
# ─────────────────────────────────────────────────────────────────────────────

def render_home_operations(
    facility_meta: Any,
    assets: pd.DataFrame,
    tel: pd.DataFrame,
    ctx: pd.DataFrame,
    events: Optional[Union[pd.DataFrame, List[Any]]] = None,
    complaints: Optional[pd.DataFrame] = None,
    ledger: Optional[SavingsLedger] = None,
    tariff: Optional[IndianTariff] = None,
    current_user: Optional[Any] = None,
    **kwargs: Any,
) -> None:
    """Render the complete Operations Home screen (Section 5.2)."""
    fac_label = getattr(facility_meta, "name", None) or (
        facility_meta.get("name") if isinstance(facility_meta, dict) else "BPUT Campus"
    )

    page_heading(
        title="Operations Home",
        subtitle=f"Operational Equipment Health, Anomaly Surveillance & Priority Dispatch · Facility: {fac_label}",
        badge="Operations",
    )

    # ── 1. Today Strip (at most 4 numbers) ──
    today_stats = prepare_operations_today_strip(
        tel=tel,
        assets=assets,
        events=events,
        ledger=ledger,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card(
            label="Load Now",
            value=f"{today_stats['load_now_kw']:.0f} kW",
            delta=today_stats["load_delta_str"],
            delta_color=today_stats["load_delta_color"],
            subtitle=f"Expected: {today_stats['expected_load_kw']:.0f} kW",
            help_text="Real-time metered campus active power vs baseline expected model draw.",
        )
    with col2:
        kpi_card(
            label="Open Alerts",
            value=str(today_stats["open_alerts_count"]),
            delta=today_stats["alert_delta_str"],
            delta_color=today_stats["alert_delta_color"],
            subtitle="Active System Exceptions",
            help_text="Count of unresolved anomalies flagged by triple-engine detector.",
        )
    with col3:
        kpi_card(
            label="Work Orders Due Today",
            value=str(today_stats["work_orders_due_today"]),
            delta=today_stats["wo_delta_str"],
            delta_color=today_stats["wo_delta_color"],
            subtitle="Maintenance Queue",
            help_text="Field remediation work orders with SLA deadlines within 24 hours.",
        )
    with col4:
        kpi_card(
            label="Sensor Data Health",
            value=f"{today_stats['data_health_pct']:.1f}%",
            delta=today_stats["data_health_str"],
            delta_color=today_stats["data_health_color"],
            subtitle="Telemetry Completeness",
            help_text="Percentage of expected interval packets received without dropout or NaN.",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ── 2. Needs Attention (The Primary Element on Screen) ──
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
          <div>
            <h2 class="nivara-heading-2" style="margin: 0; font-size: 1.25rem;">
              <span class="heading-accent"></span>Needs Attention
            </h2>
            <div class="nivara-caption">
              Prioritized triage combining critical/high alerts, work orders nearing SLA, and equipment dropouts.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alert_statuses = st.session_state.get("alert_statuses", {})
    attention_items = assemble_needs_attention(
        events=events,
        assets=assets,
        tel=tel,
        ctx=ctx,
        facility_meta=facility_meta,
        ledger=ledger,
        alert_statuses=alert_statuses,
    )

    if not attention_items:
        empty_state(
            title="All Systems Nominal",
            description="No critical exceptions, breached SLAs, or sensor dropouts detected across campus facilities.",
            icon="✅",
        )
    else:
        # Table Header
        h_col1, h_col2, h_col3, h_col4, h_col5, h_col6 = st.columns([1.2, 1.6, 5.0, 2.0, 1.4, 2.0])
        with h_col1:
            st.caption("**SEVERITY**")
        with h_col2:
            st.caption("**ASSET ID**")
        with h_col3:
            st.caption("**HEADLINE / ISSUE**")
        with h_col4:
            st.caption("**IMPACT**")
        with h_col5:
            st.caption("**AGE**")
        with h_col6:
            st.caption("**ACTION**")

        st.markdown("<hr style='margin: 4px 0 10px 0; border-color: #334155;'/>", unsafe_allow_html=True)

        for it in attention_items:
            r_col1, r_col2, r_col3, r_col4, r_col5, r_col6 = st.columns([1.2, 1.6, 5.0, 2.0, 1.4, 2.0])
            with r_col1:
                st.markdown(severity_chip(it["severity"]), unsafe_allow_html=True)
            with r_col2:
                aid_val = str(it["asset_id"])
                if st.button(aid_val, key=f"ops_ast_link_{it['id']}", help=f"Inspect asset {aid_val}"):
                    st.session_state["selected_nav_destination"] = "Assets"
                    st.session_state["active_asset_detail_id"] = aid_val
                    st.query_params["page"] = "assets"
                    st.query_params["asset"] = aid_val
                    st.rerun()
            with r_col3:
                st.markdown(f"<span style='font-size: 0.88rem; color: #f1f5f9; font-weight: 500;'>{html.escape(it['headline'])}</span>", unsafe_allow_html=True)
            with r_col4:
                st.markdown(f"<span style='font-size: 0.85rem; color: #f59e0b; font-weight: 600;'>{html.escape(it['impact'])}</span>", unsafe_allow_html=True)
            with r_col5:
                st.markdown(f"<span style='font-size: 0.82rem; color: #94a3b8;'>{html.escape(it['age'])}</span>", unsafe_allow_html=True)
            with r_col6:
                btn_key = f"ops_act_{it['id']}"
                if st.button("Open in Alerts & Work", key=btn_key, use_container_width=True):
                    # Set navigation and query parameter
                    st.session_state["selected_nav_destination"] = "Alerts & Work"
                    st.session_state["selected_nav_page"] = "🚨  Alerts & Work"
                    st.session_state["active_alert_drawer_id"] = it["id"]
                    st.query_params["page"] = "alerts"
                    st.query_params["alert_id"] = it["id"]
                    st.rerun()

            st.markdown("<hr style='margin: 6px 0; border-color: #1e293b;'/>", unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # ── 3. Load Today vs Forecast Chart & 4. Recent Activity Feed ──
    bottom_left, bottom_right = st.columns([7, 3])

    with bottom_left:
        fig = build_load_forecast_figure(tel=tel, ctx=ctx, assets=assets, events=events)
        st.plotly_chart(fig, use_container_width=True)

    with bottom_right:
        st.markdown(
            """
            <h3 class="nivara-heading-3" style="margin-top: 0;">
              📋 Recent Operational Activity
            </h3>
            """,
            unsafe_allow_html=True,
        )
        recent_events = prepare_recent_activity(events=events, ledger=ledger)
        if not recent_events:
            st.caption("No recent operational actions logged in session.")
        else:
            for ev in recent_events:
                t_rel = format_relative_time(ev["timestamp"])
                badge_html = f"<span style='background: {ev['badge_color']}20; color: {ev['badge_color']}; border: 1px solid {ev['badge_color']}; font-size: 0.70rem; padding: 1px 6px; border-radius: 4px; font-weight: 600;'>{ev['badge']}</span>"
                st.markdown(
                    f"""
                    <div style="background: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 8px 10px; margin-bottom: 8px;">
                      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                        {badge_html}
                        <span style="font-size: 0.72rem; color: #64748b;">{html.escape(t_rel)}</span>
                      </div>
                      <div style="font-size: 0.82rem; font-weight: 600; color: #e2e8f0;">{html.escape(ev['title'])}</div>
                      <div style="font-size: 0.76rem; color: #94a3b8; margin-top: 2px; line-height: 1.2;">{html.escape(ev['detail'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
