"""Assets Destination View for Nivara Campus Energy Intelligence Platform.
========================================================================
Implements Step UX-6:
- List View (Default): Searchable, filterable asset inventory with status, current draw,
  expected delta, protected-load status, and deep-linkable actions.
- Map View: 3D Mapbox GL digital twin with 2D Plotly vector schematic fallback.
  Layers: Electrical Power Density vs Thermal Comfort (ASHRAE 55 PMV).
- Asset Detail Page (?page=assets&asset=ASSET_ID):
  Full operational history, 5-minute telemetry with 3h rolling mean, shaded fault intervals,
  4-KPI operating stats ribbon, hour x weekday heatmap, peer comparison, and maintenance audit trail.
"""
from __future__ import annotations

import html
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from campus_energy.analytics.comfort import (
    get_campus_comfort_summary,
    get_comfort_snapshot,
)
from campus_energy.ui.components import (
    empty_state,
    format_inr,
    format_relative_time,
    kpi_card,
    page_heading,
    severity_chip,
    status_chip,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Colors
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "primary": "#38bdf8",     # Sky blue
    "accent": "#0ea5e9",      # Light blue
    "purple": "#8b5cf6",      # Purple
    "success": "#10b981",     # Emerald
    "warning": "#f59e0b",     # Amber
    "danger": "#ef4444",      # Rose red
    "bg_dark": "#0f172a",     # Slate dark background
    "card_bg": "#1e293b",     # Slate card surface
    "border": "#334155",      # Slate border
    "text_muted": "#94a3b8",  # Slate muted text
    "text_white": "#f8fafc",  # Crisp white text
}

CAMPUS_COORDS = {
    "BLD-A":   [84.8185, 22.2480],
    "BLD-B":   [84.8195, 22.2485],
    "BLD-ADM": [84.8192, 22.2472],
    "BLD-LIB": [84.8198, 22.2475],
    "BLD-HST": [84.8175, 22.2460],
    "BLD-UTL": [84.8210, 22.2465],
}

ZONE_OFFSETS = {
    "BLD-A-CR101": [-0.00045, -0.00025],
    "BLD-A-CR102": [-0.00030,  0.00030],
    "BLD-A-CL01":  [ 0.00035, -0.00020],
    "BLD-A-CL02":  [ 0.00040,  0.00028],
    "BLD-B-CR201": [-0.00035,  0.00020],
    "BLD-B-LAB1":  [ 0.00035, -0.00025],
    "BLD-ADM-OF101": [-0.00030, -0.00020],
    "BLD-ADM-MR201": [ 0.00030,  0.00020],
    "BLD-LIB-RH": [-0.00035,  0.00015],
    "BLD-LIB-CA": [ 0.00035, -0.00020],
    "BLD-HST-CA": [-0.00030,  0.00020],
    "BLD-HST-UA": [ 0.00030, -0.00020],
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
            ]],
        },
    }],
}

MAPBOX_TOKEN = os.environ.get("MAPBOX_ACCESS_TOKEN", "")


# ─────────────────────────────────────────────────────────────────────────────
# 1. 2D Vector Schematic Campus Map Figure (Offline / Fallback)
# ─────────────────────────────────────────────────────────────────────────────

def create_2d_campus_map_figure(
    df_map: Optional[pd.DataFrame] = None,
    is_comfort_mode: bool = False,
    campus_coords: Optional[Dict[str, List[float]]] = None,
    campus_boundary: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Generate an interactive 2D Plotly campus schematic and asset status map.

    Acts as the offline, zero-network fallback when Mapbox GL or external map tiles
    are unavailable, unreachable, or disabled.
    """
    fig = go.Figure()

    coords = campus_coords or CAMPUS_COORDS
    boundary = campus_boundary or CAMPUS_BOUNDARY

    # 1. Campus Boundary polygon
    try:
        poly_coords = boundary["features"][0]["geometry"]["coordinates"][0]
        bx = [pt[0] for pt in poly_coords]
        by = [pt[1] for pt in poly_coords]
        fig.add_trace(go.Scatter(
            x=bx,
            y=by,
            mode="lines",
            line=dict(color="#38bdf8", width=2, dash="dash"),
            name="Campus Boundary",
            hoverinfo="skip",
            showlegend=True,
        ))
    except Exception as e:
        logger.debug("Could not plot campus boundary in 2D schematic: %s", e)

    # 2. Building Footprints & Labels
    dx, dy = 0.00035, 0.00025
    bld_x, bld_y, bld_names = [], [], []
    for b_id, (b_lon, b_lat) in coords.items():
        box_x = [b_lon - dx, b_lon + dx, b_lon + dx, b_lon - dx, b_lon - dx]
        box_y = [b_lat - dy, b_lat - dy, b_lat + dy, b_lat + dy, b_lat - dy]
        fig.add_trace(go.Scatter(
            x=box_x,
            y=box_y,
            mode="lines",
            line=dict(color="#475569", width=1.5),
            fill="toself",
            fillcolor="rgba(30, 41, 59, 0.45)",
            name=f"Building {b_id}",
            hovertext=f"<b>Building {b_id} Footprint</b>",
            hoverinfo="text",
            showlegend=False,
        ))
        bld_x.append(b_lon)
        bld_y.append(b_lat + dy * 1.3)
        bld_names.append(f"<b>{b_id}</b>")

    if bld_x:
        fig.add_trace(go.Scatter(
            x=bld_x,
            y=bld_y,
            mode="text",
            text=bld_names,
            textposition="top center",
            textfont=dict(color="#94a3b8", size=11, family="Segoe UI, sans-serif"),
            name="Building Labels",
            hoverinfo="skip",
            showlegend=False,
        ))

    # 3. Assets & Sensors Scatter
    if df_map is not None and not df_map.empty:
        if is_comfort_mode:
            for cat, group in df_map.groupby("category"):
                color = group["color_hex"].iloc[0] if "color_hex" in group.columns else "#10b981"
                customdata = np.stack((
                    group["name"],
                    group["building"],
                    group["zone"],
                    group["temp_c"],
                    group["humidity"],
                    group["pmv"],
                    group["ppd"].astype(str),
                    group["advisory"],
                ), axis=-1)

                hovertemplate = (
                    "<b>%{customdata[0]}</b> (%{customdata[1]} · %{customdata[2]})<br>"
                    "🌡️ Zone Temp: <b>%{customdata[3]}</b><br>"
                    "💧 Rel Humidity: <b>%{customdata[4]}</b><br>"
                    "🎯 PMV Index: <b>%{customdata[5]}</b><br>"
                    "📊 PPD Dissatisfied: <b>%{customdata[6]}%</b><br>"
                    "💡 Advisory: <i>%{customdata[7]}</i>"
                    "<extra></extra>"
                )

                fig.add_trace(go.Scatter(
                    x=group["lon"],
                    y=group["lat"],
                    mode="markers+text",
                    marker=dict(
                        size=14,
                        color=color,
                        line=dict(width=1.5, color="#ffffff"),
                        opacity=0.92,
                    ),
                    text=group["pmv"],
                    textposition="bottom center",
                    textfont=dict(size=9, color=color),
                    name=cat,
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                ))
        else:
            alerts_df = df_map[df_map["alert"] == True] if "alert" in df_map.columns else pd.DataFrame()
            normal_df = df_map[df_map["alert"] == False] if "alert" in df_map.columns else df_map

            if not normal_df.empty:
                for cat, group in normal_df.groupby("category" if "category" in normal_df.columns else lambda _: "Meter"):
                    customdata = np.stack((
                        group["name"],
                        group.get("building", pd.Series(["Main"] * len(group))),
                        group.get("power_kw", pd.Series([0.0] * len(group))).astype(str),
                        group.get("peak_kw", pd.Series([0.0] * len(group))).astype(str),
                        group.get("day_kwh", pd.Series([0.0] * len(group))).astype(str),
                        group.get("status", pd.Series(["ACTIVE"] * len(group))),
                    ), axis=-1)
                    hovertemplate = (
                        "<b>%{customdata[0]}</b><br>"
                        "Facility: %{customdata[1]}<br>"
                        "⚡ Current Load: <b>%{customdata[2]} kW</b><br>"
                        "📈 Peak Load: %{customdata[3]} kW<br>"
                        "🔋 Today Energy: %{customdata[4]} kWh<br>"
                        "Status: %{customdata[5]}"
                        "<extra></extra>"
                    )
                    fig.add_trace(go.Scatter(
                        x=group["lon"],
                        y=group["lat"],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=group["color_hex"] if "color_hex" in group.columns else "#38bdf8",
                            line=dict(width=1, color="#ffffff"),
                            opacity=0.88,
                        ),
                        name=f"Normal: {cat}",
                        customdata=customdata,
                        hovertemplate=hovertemplate,
                    ))

            if not alerts_df.empty:
                customdata = np.stack((
                    alerts_df["name"],
                    alerts_df.get("building", pd.Series(["Main"] * len(alerts_df))),
                    alerts_df.get("power_kw", pd.Series([0.0] * len(alerts_df))).astype(str),
                    alerts_df.get("alert_type", pd.Series(["Anomaly"] * len(alerts_df))),
                    alerts_df.get("alert_desc", pd.Series(["Deviation from baseline"] * len(alerts_df))),
                ), axis=-1)
                hovertemplate = (
                    "<b>🚨 ANOMALY: %{customdata[0]}</b><br>"
                    "Building: %{customdata[1]}<br>"
                    "⚡ Load: <b>%{customdata[2]} kW</b><br>"
                    "⚠️ Type: <b>%{customdata[3]}</b><br>"
                    "ℹ️ Evidence: %{customdata[4]}"
                    "<extra></extra>"
                )
                fig.add_trace(go.Scatter(
                    x=alerts_df["lon"],
                    y=alerts_df["lat"],
                    mode="markers+text",
                    marker=dict(
                        size=18,
                        color="#ef4444",
                        symbol="circle",
                        line=dict(width=2.5, color="#fca5a5"),
                        opacity=0.95,
                    ),
                    text=alerts_df["name"].apply(lambda n: str(n).split()[-1]),
                    textposition="top center",
                    textfont=dict(size=10, color="#f87171"),
                    name="🔴 Active Anomalies",
                    customdata=customdata,
                    hovertemplate=hovertemplate,
                ))

    title_text = (
        "🌡️ 2D Campus Thermal Comfort Schematic (ASHRAE 55 — Offline Engine)"
        if is_comfort_mode
        else "🗺️ 2D Campus Asset & Telemetry Schematic (Offline Engine)"
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title=dict(text=title_text, font=dict(size=15, color="#f8fafc")),
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            title="Longitude (°E)",
            showgrid=True,
            gridcolor="#1e293b",
            zeroline=False,
            tickformat=".4f",
        ),
        yaxis=dict(
            title="Latitude (°N)",
            showgrid=True,
            gridcolor="#1e293b",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            tickformat=".4f",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11, color="#cbd5e1"),
        ),
        height=620,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Asset List Data Assembly & Pure Filter Functions
# ─────────────────────────────────────────────────────────────────────────────

def prepare_asset_list_data(
    facility_meta: Any,
    assets_df: pd.DataFrame,
    tel_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Prepare dense, enriched asset tabular records for the list view."""
    if assets_df is None or assets_df.empty:
        return pd.DataFrame()

    # Pre-parse timestamps
    tel_work = tel_df.copy() if tel_df is not None and not tel_df.empty else pd.DataFrame()
    if not tel_work.empty and not pd.api.types.is_datetime64_any_dtype(tel_work["timestamp"]):
        tel_work["timestamp"] = pd.to_datetime(tel_work["timestamp"])

    latest_global_ts = tel_work["timestamp"].max() if not tel_work.empty else None

    # Group latest telemetry by asset
    latest_by_asset: Dict[str, Dict[str, Any]] = {}
    if not tel_work.empty:
        # Sort and take last record per asset
        tel_sorted = tel_work.sort_values("timestamp")
        last_records = tel_sorted.groupby("asset_id").last().reset_index()
        for _, r in last_records.iterrows():
            latest_by_asset[str(r["asset_id"])] = {
                "power_kw": float(r["power_kw"]) if pd.notna(r.get("power_kw")) else 0.0,
                "timestamp": r["timestamp"],
            }

    # Pre-index events by asset
    active_anomalies_by_asset: Dict[str, str] = {}
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            aid = str(ev.get("asset_id", ""))
            sev = str(ev.get("severity", "medium")).lower()
            if aid:
                # Retain worst severity
                curr = active_anomalies_by_asset.get(aid, "low")
                if sev in ["critical", "high"] or curr not in ["critical", "high"]:
                    active_anomalies_by_asset[aid] = sev

    rows = []
    for _, a_row in assets_df.iterrows():
        aid = str(a_row["asset_id"])
        aname = str(a_row.get("asset_name") or aid)
        atype = str(a_row.get("asset_type") or "equipment").lower()
        bld = str(a_row.get("building_id") or "BLD-MAIN")
        zone = str(a_row.get("zone_id") or "—")
        rated_kw = float(a_row.get("rated_power_kw", 0.0)) if pd.notna(a_row.get("rated_power_kw")) else 0.0

        # Protection check
        is_prot = False
        prot_cat = ""
        prot_reason = ""
        min_pwr_kw = None
        if hasattr(facility_meta, "is_asset_protected"):
            is_prot = bool(facility_meta.is_asset_protected(aid))
            if is_prot and hasattr(facility_meta, "get_protected_load"):
                p_load = facility_meta.get_protected_load(aid)
                if p_load:
                    prot_cat = getattr(p_load, "category", "")
                    prot_reason = getattr(p_load, "reason", "")
                    min_pwr_kw = getattr(p_load, "min_power_kw", None)
        elif isinstance(facility_meta, dict) and "protected_loads" in facility_meta:
            is_prot = aid in facility_meta["protected_loads"]

        # Telemetry & Expected Load calculation
        tel_info = latest_by_asset.get(aid)
        if tel_info:
            current_kw = tel_info["power_kw"]
            last_ts = tel_info["timestamp"]
            rel_last_seen = format_relative_time(last_ts, reference_dt=latest_global_ts)
        else:
            current_kw = 0.0
            last_ts = None
            rel_last_seen = "No Telemetry"

        # Baseline expected load
        expected_kw = rated_kw * 0.6 if rated_kw > 0 else (current_kw if current_kw > 0 else 5.0)
        delta_kw = current_kw - expected_kw
        delta_pct = (delta_kw / max(expected_kw, 0.1)) * 100.0

        # Status determination: Critical, Elevated, Normal
        anom_sev = active_anomalies_by_asset.get(aid)
        if anom_sev in ["critical", "high"]:
            status = "Critical"
        elif anom_sev in ["medium", "low"]:
            status = "Elevated"
        elif abs(delta_pct) > 40.0 and current_kw > 1.0:
            status = "Elevated"
        elif current_kw == 0.0 and expected_kw > 2.0 and atype != "meter":
            status = "Elevated"
        else:
            status = "Normal"

        rows.append({
            "asset_id": aid,
            "asset_name": aname,
            "asset_type": atype,
            "building_id": bld,
            "zone_id": zone,
            "rated_power_kw": rated_kw,
            "equipment_count": a_row.get("equipment_count", 1),
            "monitoring_level": a_row.get("monitoring_level", "equipment"),
            "current_kw": round(current_kw, 2),
            "expected_kw": round(expected_kw, 2),
            "delta_kw": round(delta_kw, 2),
            "delta_pct": round(delta_pct, 1),
            "status": status,
            "is_protected": is_prot,
            "prot_category": prot_cat,
            "prot_reason": prot_reason,
            "min_power_kw": min_pwr_kw,
            "last_seen_ts": last_ts,
            "relative_last_seen": rel_last_seen,
        })

    return pd.DataFrame(rows)


def filter_assets(
    df: pd.DataFrame,
    search_query: str = "",
    building: str = "All",
    asset_type: str = "All",
    protection: str = "All",
    status: str = "All",
) -> pd.DataFrame:
    """Filter prepared assets DataFrame by multiple criteria."""
    if df is None or df.empty:
        return pd.DataFrame()

    res = df.copy()

    # Search filter
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        mask = (
            res["asset_id"].str.lower().str.contains(q, na=False)
            | res["asset_name"].str.lower().str.contains(q, na=False)
            | res["building_id"].str.lower().str.contains(q, na=False)
            | res["zone_id"].str.lower().str.contains(q, na=False)
            | res["asset_type"].str.lower().str.contains(q, na=False)
        )
        res = res[mask]

    # Building filter
    if building != "All":
        res = res[res["building_id"] == building]

    # Asset Type filter
    if asset_type != "All":
        res = res[res["asset_type"] == asset_type]

    # Protection filter
    if protection == "Protected Only":
        res = res[res["is_protected"] == True]
    elif protection == "Unprotected":
        res = res[res["is_protected"] == False]

    # Status filter
    if status != "All":
        res = res[res["status"].str.lower() == status.lower()]

    return res


# ─────────────────────────────────────────────────────────────────────────────
# 3. Asset Detail Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_asset_detail_data(
    asset_id: str,
    facility_meta: Any,
    assets_df: pd.DataFrame,
    tel_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    ledger: Optional[Any] = None,
) -> Dict[str, Any]:
    """Assemble all telemetry traces, statistics, peer comparison, and audit history for an asset."""
    # Find asset row
    match = assets_df[assets_df["asset_id"] == asset_id] if assets_df is not None and not assets_df.empty else pd.DataFrame()
    if match.empty:
        row = {
            "asset_id": asset_id,
            "asset_name": asset_id,
            "asset_type": "equipment",
            "building_id": "BLD-MAIN",
            "zone_id": "—",
            "rated_power_kw": 10.0,
            "equipment_count": 1,
            "monitoring_level": "equipment",
        }
    else:
        row = match.iloc[0].to_dict()

    # Protection status
    is_prot = False
    prot_cat = ""
    prot_reason = ""
    min_power_kw = None
    if hasattr(facility_meta, "is_asset_protected"):
        is_prot = bool(facility_meta.is_asset_protected(asset_id))
        if is_prot and hasattr(facility_meta, "get_protected_load"):
            p_load = facility_meta.get_protected_load(asset_id)
            if p_load:
                prot_cat = getattr(p_load, "category", "mission_critical")
                prot_reason = getattr(p_load, "reason", "Non-curtailable critical load")
                min_power_kw = getattr(p_load, "min_power_kw", None)

    # Telemetry slice
    a_tel = pd.DataFrame()
    if tel_df is not None and not tel_df.empty:
        a_tel = tel_df[tel_df["asset_id"] == asset_id].copy()
        if not a_tel.empty:
            if not pd.api.types.is_datetime64_any_dtype(a_tel["timestamp"]):
                a_tel["timestamp"] = pd.to_datetime(a_tel["timestamp"])
            a_tel = a_tel.sort_values("timestamp")
            # 3h rolling mean (36 5-min intervals)
            a_tel["rolling_3h_kw"] = a_tel["power_kw"].rolling(36, min_periods=1).mean()

    # Operating KPIs
    if not a_tel.empty:
        mean_kw = float(a_tel["power_kw"].mean())
        peak_kw = float(a_tel["power_kw"].max())
        nonzero_tel = a_tel[a_tel["power_kw"] > 0.05]
        standby_kw = float(nonzero_tel["power_kw"].quantile(0.05)) if not nonzero_tel.empty else float(a_tel["power_kw"].min())
        total_kwh = float(a_tel["interval_energy_kwh"].sum()) if "interval_energy_kwh" in a_tel.columns else mean_kw * len(a_tel) * (5.0 / 60.0)
    else:
        mean_kw = float(row.get("rated_power_kw", 0.0) or 5.0) * 0.5
        peak_kw = float(row.get("rated_power_kw", 0.0) or 10.0)
        standby_kw = mean_kw * 0.2
        total_kwh = 0.0

    # Diurnal Matrix Heatmap (Day of Week x Hour)
    heatmap_pivot = pd.DataFrame()
    if not a_tel.empty:
        a_tel["hour"] = a_tel["timestamp"].dt.hour
        a_tel["dow"] = a_tel["timestamp"].dt.day_name()
        hm_df = a_tel.groupby(["dow", "hour"])["power_kw"].mean().reset_index()
        heatmap_pivot = hm_df.pivot(index="dow", columns="hour", values="power_kw").fillna(0.0)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])

    # Peer comparison (same building, same type)
    peer_stats = pd.DataFrame()
    if assets_df is not None and not assets_df.empty and not a_tel.empty:
        bld = row.get("building_id")
        atype = row.get("asset_type")
        sibling_ids = assets_df[
            (assets_df["building_id"] == bld)
            & (assets_df["asset_type"] == atype)
        ]["asset_id"].tolist()

        if len(sibling_ids) > 1 and tel_df is not None:
            peer_tel = tel_df[tel_df["asset_id"].isin(sibling_ids)]
            if not peer_tel.empty:
                peer_grp = peer_tel.groupby("asset_id")["power_kw"].mean().reset_index()
                peer_grp.columns = ["asset_id", "avg_power_kw"]
                peer_grp["is_selected"] = peer_grp["asset_id"] == asset_id
                # Map asset names
                id_to_name = dict(zip(assets_df["asset_id"], assets_df["asset_name"]))
                peer_grp["asset_name"] = peer_grp["asset_id"].map(lambda x: id_to_name.get(x, x))
                peer_stats = peer_grp.sort_values("avg_power_kw", ascending=False)

    # Shaded fault intervals & anomalies
    asset_events = pd.DataFrame()
    if events_df is not None and not events_df.empty:
        asset_events = events_df[events_df["asset_id"] == asset_id].copy()

    # Asset Operational History (combining anomaly alerts + maintenance work orders)
    audit_history = []
    if not asset_events.empty:
        for _, ev in asset_events.iterrows():
            audit_history.append({
                "timestamp": str(ev.get("start_time") or ev.get("detected_at", "")),
                "record_type": "Anomaly Alert",
                "id": str(ev.get("event_id") or "ALT-EVT"),
                "title": str(ev.get("event_type", "Operational Anomaly")).replace("_", " ").title(),
                "severity": str(ev.get("severity", "medium")).lower(),
                "status": "active" if pd.isna(ev.get("end_time")) else "done",
                "details": str(ev.get("description") or ev.get("root_cause") or "Telemetry baseline variance"),
                "sla_impact": "Immediate Attention Required" if str(ev.get("severity")).lower() in ["critical", "high"] else "Under Observation",
            })

    # Pull work orders from ledger targeting this asset
    if ledger is not None and hasattr(ledger, "get_items"):
        try:
            items = ledger.get_items()
            for wo in items:
                rec = getattr(wo, "recommendation", None)
                target = getattr(rec, "target", "") if rec else ""
                if target == asset_id or asset_id in target:
                    audit_history.append({
                        "timestamp": wo.created_at.strftime("%Y-%m-%d %H:%M") if hasattr(wo.created_at, "strftime") else str(wo.created_at),
                        "record_type": "Work Order",
                        "id": wo.work_order_id,
                        "title": getattr(rec, "action_text", "Field Service Work Order") if rec else "Equipment Remediation",
                        "severity": getattr(wo, "priority", "medium").lower(),
                        "status": getattr(wo, "status", "open"),
                        "details": f"Assigned to: {getattr(wo, 'assignee', 'Facilities')} · MV: {getattr(wo, 'mv_method', 'IPMVP')}",
                        "sla_impact": f"SLA: {wo.sla_deadline.strftime('%b %d %H:%M')}" if hasattr(wo, 'sla_deadline') and hasattr(wo.sla_deadline, 'strftime') else "Standard SLA",
                    })
        except Exception as e:
            logger.debug("Failed querying ledger work orders: %s", e)

    # Sort history newest first
    audit_history.sort(key=lambda x: str(x.get("timestamp", "")), reverse=True)

    return {
        "asset_row": row,
        "is_protected": is_prot,
        "prot_category": prot_cat,
        "prot_reason": prot_reason,
        "min_power_kw": min_power_kw,
        "telemetry": a_tel,
        "mean_kw": mean_kw,
        "peak_kw": peak_kw,
        "standby_kw": standby_kw,
        "total_kwh": total_kwh,
        "heatmap_pivot": heatmap_pivot,
        "peer_stats": peer_stats,
        "events": asset_events,
        "audit_history": audit_history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. View Modes: List View, Map View, Asset Detail Page
# ─────────────────────────────────────────────────────────────────────────────

def render_asset_list_view(
    prepared_df: pd.DataFrame,
    facility_meta: Any,
) -> None:
    """Render the primary dense, searchable list view for assets."""
    if prepared_df.empty:
        empty_state(
            title="No Assets Found",
            message="No equipment or meter assets registered for this facility.",
            action_label="Refresh",
        )
        return

    # Filter Bar
    c_search, c_bld, c_type, c_prot, c_stat = st.columns([3, 1.8, 1.8, 1.8, 1.8])

    with c_search:
        search_q = st.text_input(
            "Search Assets",
            placeholder="Search by ID, name, building, zone, type...",
            key="assets_list_search_input",
            label_visibility="collapsed",
        )

    with c_bld:
        buildings = ["All"] + sorted(prepared_df["building_id"].unique().tolist())
        sel_bld = st.selectbox(
            "Building",
            buildings,
            key="assets_list_bld_select",
            label_visibility="collapsed",
        )

    with c_type:
        types = ["All"] + sorted(prepared_df["asset_type"].unique().tolist())
        sel_type = st.selectbox(
            "Asset Type",
            types,
            key="assets_list_type_select",
            label_visibility="collapsed",
        )

    with c_prot:
        sel_prot = st.selectbox(
            "Protection",
            ["All", "Protected Only", "Unprotected"],
            key="assets_list_prot_select",
            label_visibility="collapsed",
        )

    with c_stat:
        sel_stat = st.selectbox(
            "Status",
            ["All", "Normal", "Elevated", "Critical"],
            key="assets_list_stat_select",
            label_visibility="collapsed",
        )

    # Apply filters
    filtered = filter_assets(
        prepared_df,
        search_query=search_q,
        building=sel_bld,
        asset_type=sel_type,
        protection=sel_prot,
        status=sel_stat,
    )

    # Count badge
    total_count = len(prepared_df)
    shown_count = len(filtered)
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 8px 0 12px 0;">
            <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 500;">
                Showing <b style="color: #f8fafc;">{shown_count}</b> of {total_count} assets
            </span>
            <span style="font-size: 0.78rem; color: #64748b;">
                Click <b>View Details</b> on any asset to inspect power curves, peer benchmarking & audit trail
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("🔍 No assets match the selected filters. Clear search or adjust dropdowns.")
        return

    # Table Header Row
    h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = st.columns([1.6, 2.4, 1.2, 1.5, 1.3, 1.3, 1.5, 1.4, 1.4, 1.4])
    with h1:
        st.caption("**ASSET ID**")
    with h2:
        st.caption("**NAME & TYPE**")
    with h3:
        st.caption("**BLD / ZONE**")
    with h4:
        st.caption("**STATUS**")
    with h5:
        st.caption("**CURRENT kW**")
    with h6:
        st.caption("**vs EXPECTED**")
    with h7:
        st.caption("**PROTECTION**")
    with h8:
        st.caption("**RATED CAP**")
    with h9:
        st.caption("**LAST SEEN**")
    with h10:
        st.caption("**ACTION**")

    st.markdown("<hr style='margin: 4px 0 10px 0; border-color: #334155;'/>", unsafe_allow_html=True)

    # Render tabular asset rows
    for _, row in filtered.iterrows():
        aid = row["asset_id"]
        r1, r2, r3, r4, r5, r6, r7, r8, r9, r10 = st.columns([1.6, 2.4, 1.2, 1.5, 1.3, 1.3, 1.5, 1.4, 1.4, 1.4])

        with r1:
            st.markdown(f"<code style='color: #38bdf8; font-weight: 600; font-size: 0.82rem;'>{html.escape(aid)}</code>", unsafe_allow_html=True)

        with r2:
            st.markdown(
                f"""
                <div style="line-height: 1.25;">
                    <div style="font-size: 0.86rem; font-weight: 600; color: #f1f5f9;">{html.escape(row['asset_name'])}</div>
                    <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">{html.escape(row['asset_type'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r3:
            st.markdown(
                f"""
                <div style="font-size: 0.82rem; color: #cbd5e1;">
                    <b>{html.escape(row['building_id'])}</b>
                    <div style="font-size: 0.72rem; color: #64748b;">{html.escape(row['zone_id'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r4:
            # Status chip
            stat = row["status"]
            if stat == "Critical":
                st.markdown(severity_chip("critical"), unsafe_allow_html=True)
            elif stat == "Elevated":
                st.markdown(severity_chip("high"), unsafe_allow_html=True)
            else:
                st.markdown('<span class="chip chip-status chip-status-verified"><span class="chip-dot"></span>Normal</span>', unsafe_allow_html=True)

        with r5:
            st.markdown(f"<span style='font-size: 0.88rem; font-weight: 700; color: #f8fafc;'>{row['current_kw']:.2f} kW</span>", unsafe_allow_html=True)

        with r6:
            delta_kw = row["delta_kw"]
            delta_pct = row["delta_pct"]
            d_color = "#ef4444" if delta_kw > 2.0 else ("#10b981" if delta_kw <= 0.5 else "#f59e0b")
            d_sign = "+" if delta_kw > 0 else ""
            st.markdown(
                f"<span style='font-size: 0.80rem; font-weight: 600; color: {d_color};'>{d_sign}{delta_kw:.1f} kW ({d_sign}{delta_pct:.0f}%)</span>",
                unsafe_allow_html=True,
            )

        with r7:
            if row["is_protected"]:
                st.markdown(
                    f"""
                    <span style="background: rgba(59, 130, 246, 0.2); border: 1px solid #3b82f6; color: #60a5fa; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;" title="{html.escape(row['prot_reason'])}">
                        🛡️ Protected
                    </span>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown("<span style='font-size: 0.76rem; color: #64748b;'>Flexible</span>", unsafe_allow_html=True)

        with r8:
            cap_str = f"{row['rated_power_kw']:.1f} kW" if row["rated_power_kw"] > 0 else "—"
            st.markdown(f"<span style='font-size: 0.80rem; color: #cbd5e1;'>{cap_str}</span>", unsafe_allow_html=True)

        with r9:
            st.markdown(f"<span style='font-size: 0.78rem; color: #94a3b8;'>{html.escape(row['relative_last_seen'])}</span>", unsafe_allow_html=True)

        with r10:
            if st.button("View Details →", key=f"btn_asset_det_{aid}", use_container_width=True):
                st.session_state["active_asset_detail_id"] = aid
                st.query_params["page"] = "assets"
                st.query_params["asset"] = aid
                st.rerun()

        st.markdown("<hr style='margin: 6px 0; border-color: #1e293b;'/>", unsafe_allow_html=True)


def render_asset_map_view(
    facility_meta: Any,
    assets_df: pd.DataFrame,
    tel_df: pd.DataFrame,
    ctx_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
) -> None:
    """Render the Map View with Mapbox GL satellite and Plotly 2D fallback."""
    now_real = datetime.now()
    real_time_str = now_real.strftime("%Y-%m-%d %H:%M")
    real_date_str = now_real.strftime("%Y-%m-%d")

    latest_ts = tel_df["timestamp"].max() if tel_df is not None and not tel_df.empty else datetime.now()
    if isinstance(latest_ts, str):
        latest_ts = pd.to_datetime(latest_ts)
    latest_date = latest_ts.date() if hasattr(latest_ts, "date") else datetime.now().date()

    latest_day_tel = tel_df[tel_df["timestamp"].dt.date == latest_date] if tel_df is not None and not tel_df.empty else pd.DataFrame()
    latest_tel = tel_df[tel_df["timestamp"] == latest_ts] if tel_df is not None and not tel_df.empty else pd.DataFrame()

    # Pre-index events by asset
    alert_map_by_asset: Dict[str, Any] = {}
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            aid = str(ev.get("asset_id", ""))
            if aid:
                alert_map_by_asset[aid] = ev

    # Comfort snapshot
    comfort_snapshot_df = pd.DataFrame()
    comfort_summary = {"avg_temp": 24.5, "avg_humidity": 55.0, "avg_pmv": 0.1, "pct_compliant": 85.0}
    try:
        comfort_snapshot_df = get_comfort_snapshot(
            ctx_df, assets_df, latest_ts, tel_df=latest_day_tel,
            campus_coords=CAMPUS_COORDS, zone_offsets=ZONE_OFFSETS
        )
        comfort_summary = get_campus_comfort_summary(comfort_snapshot_df)
    except Exception as e:
        logger.debug("Could not calculate comfort snapshot: %s", e)

    # Map Controls
    map_col, ctrl_col = st.columns([3, 1])

    with ctrl_col:
        view_mode = st.radio(
            "Map Overlay Mode",
            ["⚡ Electrical Power Density (kW)", "🌡️ Thermal Comfort (ASHRAE 55 PMV)"],
            index=0,
            help="Toggle between real-time electrical telemetry and ASHRAE 55 thermal comfort.",
            key="assets_map_overlay_mode",
        )
        is_comfort_mode = "Thermal Comfort" in view_mode

        time_mode = st.radio(
            "Clock Anchor",
            ["🔴 Live Real-Time (Today)", "📅 Simulation Dataset Date"],
            index=0,
            key="assets_map_time_mode",
        )
        display_date_str = real_date_str if "Live" in time_mode else str(latest_date)
        display_time_str = real_time_str if "Live" in time_mode else latest_ts.strftime('%Y-%m-%d %H:%M')

        is_valid_mapbox = bool(
            MAPBOX_TOKEN
            and MAPBOX_TOKEN.startswith("pk.")
            and len(MAPBOX_TOKEN) > 30
            and os.environ.get("NIVARA_OFFLINE_MAP") != "1"
        )
        map_engine = st.selectbox(
            "Map Engine",
            ["3D Mapbox GL (Satellite)", "2D Plotly Schematic (Offline Fallback)"],
            index=0 if is_valid_mapbox else 1,
            key="assets_map_engine_select",
        )

        map_style = st.selectbox("Map Style", ["Satellite + Streets", "Dark", "Light"], key="assets_map_style_select")

        # Direct asset selector to deep-link
        st.divider()
        st.markdown("**Inspect Asset Detail**")
        all_asset_ids = sorted(assets_df["asset_id"].unique().tolist()) if assets_df is not None and not assets_df.empty else []
        chosen_inspect_asset = st.selectbox("Select Asset to Open Detail Page", all_asset_ids, key="assets_map_inspect_select")
        if st.button("Open Asset Detail Page →", key="assets_map_inspect_btn", use_container_width=True):
            st.session_state["active_asset_detail_id"] = chosen_inspect_asset
            st.query_params["page"] = "assets"
            st.query_params["asset"] = chosen_inspect_asset
            st.rerun()

    # Build Map Data
    map_data = []
    if not is_comfort_mode:
        # Building Main Meters
        meter_tel = latest_tel[latest_tel["reading_type"] == "meter"] if not latest_tel.empty and "reading_type" in latest_tel.columns else pd.DataFrame()
        for _, row in meter_tel.iterrows():
            bld_id = str(row["asset_id"]).replace("MTR-", "")
            if bld_id not in CAMPUS_COORDS:
                continue
            power = float(row["power_kw"]) if pd.notna(row["power_kw"]) else 0.0

            bld_day = latest_day_tel[latest_day_tel["asset_id"] == row["asset_id"]] if not latest_day_tel.empty else pd.DataFrame()
            day_kwh = float(bld_day["interval_energy_kwh"].sum()) if not bld_day.empty and "interval_energy_kwh" in bld_day.columns else 0.0
            peak_kw = float(bld_day["power_kw"].max()) if not bld_day.empty else power

            bld_alert = alert_map_by_asset.get(row["asset_id"]) or alert_map_by_asset.get(bld_id)
            has_alert = bld_alert is not None

            is_bld_prot = hasattr(facility_meta, "is_asset_protected") and facility_meta.is_asset_protected(row["asset_id"])
            bld_prot_load = facility_meta.get_protected_load(row["asset_id"]) if is_bld_prot and hasattr(facility_meta, "get_protected_load") else None

            color = [239, 68, 68, 255] if has_alert else ([16, 185, 129, 230] if power < 5.0 else [245, 158, 11, 230])

            map_data.append({
                "id": row["asset_id"],
                "name": f"Building {bld_id} Main Incomer",
                "category": "Building Main Meters",
                "building": bld_id,
                "zone": "Whole Facility",
                "lon": CAMPUS_COORDS[bld_id][0],
                "lat": CAMPUS_COORDS[bld_id][1],
                "power_kw": round(power, 2),
                "day_kwh": round(day_kwh, 1),
                "peak_kw": round(peak_kw, 2),
                "temp_c": "N/A",
                "status": "OFF" if power == 0 else "ACTIVE",
                "color": color,
                "color_hex": "#ef4444" if has_alert else "#10b981",
                "dot_size": 24,
                "is_meter": True,
                "is_comfort": False,
                "is_protected": is_bld_prot,
                "prot_category": getattr(bld_prot_load, "category", "") if bld_prot_load else "",
                "prot_reason": getattr(bld_prot_load, "reason", "") if bld_prot_load else "",
                "alert": has_alert,
                "alert_type": "Critical Alarm" if has_alert else "",
                "alert_desc": "Operational deviation" if has_alert else "Optimal baseline conformance",
            })

        # Sub-meters & Equipment
        equip_tel = latest_tel[latest_tel["reading_type"] == "equipment"] if not latest_tel.empty and "reading_type" in latest_tel.columns else pd.DataFrame()
        for _, row in equip_tel.iterrows():
            a_id = str(row["asset_id"])
            a_match = assets_df[assets_df["asset_id"] == a_id] if assets_df is not None else pd.DataFrame()
            if a_match.empty:
                continue
            a_info = a_match.iloc[0]
            bld_id = a_info["building_id"]
            z_id = a_info.get("zone_id")

            if bld_id not in CAMPUS_COORDS:
                continue

            base_lon, base_lat = CAMPUS_COORDS[bld_id]
            z_offset = ZONE_OFFSETS.get(z_id, [0.0, 0.0])
            hash_val = abs(hash(a_id)) % 100
            jitter_lon = (hash_val / 100000.0) - 0.0005
            jitter_lat = ((abs(hash(a_id[::-1])) % 100) / 100000.0) - 0.0005
            lon = base_lon + z_offset[0] + jitter_lon
            lat = base_lat + z_offset[1] + jitter_lat

            power = float(row["power_kw"]) if pd.notna(row.get("power_kw")) else 0.0
            state = str(row.get("state") or ("ON" if power > 0.05 else "OFF"))
            temp_val = f"{row['temperature_c']:.1f} °C" if pd.notna(row.get("temperature_c")) and row["temperature_c"] > 0 else "N/A"

            eq_day = latest_day_tel[latest_day_tel["asset_id"] == a_id] if not latest_day_tel.empty else pd.DataFrame()
            day_kwh = float(eq_day["interval_energy_kwh"].sum()) if not eq_day.empty and "interval_energy_kwh" in eq_day.columns else 0.0
            peak_kw = float(eq_day["power_kw"].max()) if not eq_day.empty else power

            eq_alert = alert_map_by_asset.get(a_id)
            has_alert = eq_alert is not None

            is_eq_prot = hasattr(facility_meta, "is_asset_protected") and facility_meta.is_asset_protected(a_id)
            eq_prot_load = facility_meta.get_protected_load(a_id) if is_eq_prot and hasattr(facility_meta, "get_protected_load") else None

            color = [239, 68, 68, 255] if has_alert else ([56, 189, 248, 220] if power > 0.5 else [100, 116, 139, 180])

            map_data.append({
                "id": a_id,
                "name": a_info["asset_name"],
                "category": a_info.get("asset_type", "Equipment"),
                "building": bld_id,
                "zone": z_id or "Central Area",
                "lon": lon,
                "lat": lat,
                "power_kw": round(power, 3),
                "day_kwh": round(day_kwh, 2),
                "peak_kw": round(peak_kw, 3),
                "temp_c": temp_val,
                "status": state,
                "color": color,
                "color_hex": "#ef4444" if has_alert else "#38bdf8",
                "dot_size": 14,
                "is_meter": False,
                "is_comfort": False,
                "is_protected": is_eq_prot,
                "prot_category": getattr(eq_prot_load, "category", "") if eq_prot_load else "",
                "prot_reason": getattr(eq_prot_load, "reason", "") if eq_prot_load else "",
                "alert": has_alert,
                "alert_type": "Elevated Draw" if has_alert else "",
                "alert_desc": "Observed anomaly" if has_alert else "Normal reading",
            })
    else:
        # Comfort data
        if not comfort_snapshot_df.empty:
            HEX_TO_RGB = {
                "#10b981": [16, 185, 129, 240],
                "#f59e0b": [245, 158, 11, 240],
                "#ef4444": [239, 68, 68, 255],
                "#38bdf8": [56, 189, 248, 240],
                "#6366f1": [99, 102, 241, 240],
            }
            for _, r in comfort_snapshot_df.iterrows():
                pmv_val = float(r["pmv"])
                c_hex = r["color"]
                c_rgb = HEX_TO_RGB.get(c_hex, [16, 185, 129, 240])
                map_data.append({
                    "id": r["zone_id"],
                    "name": r["zone_name"],
                    "category": r["category"],
                    "building": r["building_id"],
                    "zone": r["zone_id"],
                    "lon": float(r["lon"]),
                    "lat": float(r["lat"]),
                    "temp_c": f"{r['temperature_c']:.1f} °C",
                    "humidity": f"{r['humidity_pct']:.1f}%",
                    "pmv": f"{pmv_val:+.2f}",
                    "ppd": float(r["ppd"]),
                    "advisory": r["advisory"],
                    "color": c_rgb,
                    "color_hex": c_hex,
                    "dot_size": 20,
                    "is_meter": False,
                    "is_comfort": True,
                    "is_stress": abs(pmv_val) > 1.5,
                    "alert": abs(pmv_val) > 1.5,
                })

    df_map = pd.DataFrame(map_data)

    with map_col:
        use_2d_fallback = (map_engine == "2D Plotly Schematic (Offline Fallback)") or (not is_valid_mapbox)
        if use_2d_fallback:
            if not is_valid_mapbox and map_engine == "3D Mapbox GL (Satellite)":
                st.info("ℹ️ Mapbox API key not configured or offline mode enabled. Seamlessly displaying interactive 2D Plotly campus schematic.")
            fig_2d = create_2d_campus_map_figure(df_map, is_comfort_mode=is_comfort_mode)
            st.plotly_chart(fig_2d, use_container_width=True)
        else:
            style_url = {
                "Satellite + Streets": "mapbox://styles/mapbox/satellite-streets-v12",
                "Dark": "mapbox://styles/mapbox/dark-v11",
                "Light": "mapbox://styles/mapbox/light-v11",
            }.get(map_style, "mapbox://styles/mapbox/satellite-streets-v12")

            bld_poly_json = json.dumps({"type": "FeatureCollection", "features": []})
            map_json = df_map.to_json(orient="records") if not df_map.empty else "[]"
            bnd_json = json.dumps(CAMPUS_BOUNDARY)

            html_code = f"""
            <!DOCTYPE html><html><head>
            <meta charset="utf-8"><meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no">
            <link href="https://api.mapbox.com/mapbox-gl-js/v3.0.0/mapbox-gl.css" rel="stylesheet">
            <script src="https://api.mapbox.com/mapbox-gl-js/v3.0.0/mapbox-gl.js"></script>
            <style>
              body {{ margin:0; background:#0f172a; }}
              #map {{ position:absolute; top:0; bottom:0; width:100%; border-radius:8px; }}
              .mapboxgl-popup-content {{ background:#0f172a; color:#f8fafc; border-radius:10px; padding:14px; font-family:'Segoe UI', sans-serif; font-size:12px; border:1px solid #334155; }}
            </style>
            </head><body>
            <div id="map"></div>
            <script>
            mapboxgl.accessToken = '{MAPBOX_TOKEN}';
            const map = new mapboxgl.Map({{
              container:'map', style:'{style_url}',
              center:[84.8192,22.2472], zoom:16.4, pitch:58, bearing:-15, antialias:true
            }});
            map.addControl(new mapboxgl.NavigationControl(), 'top-left');
            </script>
            </body></html>
            """
            try:
                components.html(html_code, height=620)
            except Exception as e:
                st.warning(f"⚠️ 3D Mapbox rendering error ({e}). Showing 2D vector schematic fallback.")
                fig_2d = create_2d_campus_map_figure(df_map, is_comfort_mode=is_comfort_mode)
                st.plotly_chart(fig_2d, use_container_width=True)


def render_asset_detail_page(
    asset_id: str,
    facility_meta: Any,
    assets_df: pd.DataFrame,
    tel_df: pd.DataFrame,
    ctx_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    complaints_df: Optional[pd.DataFrame] = None,
    ledger: Optional[Any] = None,
) -> None:
    """Render the dedicated Asset Detail Page with full telemetry history and audit trail."""
    data = prepare_asset_detail_data(
        asset_id=asset_id,
        facility_meta=facility_meta,
        assets_df=assets_df,
        tel_df=tel_df,
        events_df=events_df,
        ledger=ledger,
    )

    row = data["asset_row"]
    is_prot = data["is_protected"]
    a_tel = data["telemetry"]

    # "← Back to Assets List" link button
    col_back, col_actions = st.columns([6, 2])
    with col_back:
        if st.button("← Back to Assets List", key="btn_back_to_assets_list"):
            st.session_state.pop("active_asset_detail_id", None)
            if "asset" in st.query_params:
                del st.query_params["asset"]
            st.query_params["page"] = "assets"
            st.rerun()

    # Asset Header
    page_heading(
        title=f"Asset Detail · {row.get('asset_name', asset_id)}",
        subtitle=f"Equipment Tag: {asset_id} · Facility: {row.get('building_id')} · Zone: {row.get('zone_id', '—')}",
        badge=str(row.get("asset_type", "Equipment")).upper(),
    )

    # Mission-Critical Protected Load Badge if applicable
    if is_prot:
        prot_cat = data.get("prot_category", "life_safety").replace("_", " ").title()
        min_pwr = f"Min Required Baseload: <b>{data.get('min_power_kw')} kW</b> · " if data.get("min_power_kw") else ""
        st.markdown(
            f"""
            <div style="background: rgba(30, 58, 138, 0.35); border: 1.5px solid #3b82f6; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #60a5fa; font-size: 1.0rem;">
                        🛡️ Mission-Critical Protected Load (Non-Curtailable)
                    </span>
                    <span style="background: #1e40af; color: #bfdbfe; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                        Category: {prot_cat}
                    </span>
                </div>
                <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 6px; line-height: 1.45;">
                    {min_pwr}<b>Operational Rationale:</b> {html.escape(data.get('prot_reason') or 'Essential clinical / continuous operation asset.')}
                    <br/><span style="color: #93c5fd; font-size: 0.78rem;">Service-Safe Guard strictly prevents automated load-shedding or aggressive setpoint setbacks on this asset.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Operating Stats Ribbon: 4 kpi_cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card(
            label="Mean Power",
            value=f"{data['mean_kw']:.2f} kW",
            subtitle="24h Average Draw",
        )
    with k2:
        kpi_card(
            label="Peak Demand",
            value=f"{data['peak_kw']:.2f} kW",
            subtitle="Max Interval Recorded",
        )
    with k3:
        kpi_card(
            label="Standby Baseload",
            value=f"{data['standby_kw']:.2f} kW",
            subtitle="P5 Operating Draw",
        )
    with k4:
        kpi_card(
            label="Total Energy",
            value=f"{data['total_kwh']:,.1f} kWh",
            subtitle="Interval Aggregation",
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # Power Consumption Trace
    st.subheader("⚡ Power Consumption Trace (5-Minute Telemetry)")
    if a_tel.empty:
        st.info(f"No telemetry records found for asset {asset_id}.")
    else:
        fig_trace = go.Figure()

        # Telemetry line
        fig_trace.add_trace(go.Scatter(
            x=a_tel["timestamp"],
            y=a_tel["power_kw"],
            mode="lines",
            name="Power (kW)",
            line=dict(color=COLORS["primary"], width=1.8),
            hovertemplate="<b>%{x|%b %d %H:%M}</b><br>Power: %{y:.2f} kW<extra></extra>",
        ))

        # 3h rolling mean
        if "rolling_3h_kw" in a_tel.columns:
            fig_trace.add_trace(go.Scatter(
                x=a_tel["timestamp"],
                y=a_tel["rolling_3h_kw"],
                mode="lines",
                name="3h Rolling Mean",
                line=dict(color=COLORS["warning"], width=2.2, dash="dot"),
                hovertemplate="3h Mean: %{y:.2f} kW<extra></extra>",
            ))

        # Shaded fault intervals
        events_for_asset = data.get("events")
        if events_for_asset is not None and not events_for_asset.empty:
            for _, ev in events_for_asset.iterrows():
                s_time = ev.get("start_time")
                e_time = ev.get("end_time")
                if pd.notna(s_time) and pd.notna(e_time):
                    fig_trace.add_vrect(
                        x0=s_time,
                        x1=e_time,
                        fillcolor="rgba(239, 68, 68, 0.16)",
                        line_width=1,
                        line_color="rgba(239, 68, 68, 0.5)",
                        annotation_text=str(ev.get("event_type", "fault")).replace("_", " "),
                        annotation_position="top left",
                        annotation_font_color="#ef4444",
                    )

        fig_trace.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font_color="#94a3b8",
            xaxis=dict(title="Timestamp", gridcolor="#1e293b"),
            yaxis=dict(title="Power (kW)", gridcolor="#1e293b"),
            hovermode="x unified",
            legend=dict(bgcolor="#1e293b", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_trace, use_container_width=True)

    # Two-Column Section: Heatmap & Peer Comparison
    col_hm, col_peer = st.columns([1, 1])

    with col_hm:
        st.subheader("🗓️ Hour × Weekday Operating Heatmap")
        hm_pivot = data.get("heatmap_pivot")
        if hm_pivot is not None and not hm_pivot.empty:
            fig_hm = px.imshow(
                hm_pivot,
                aspect="auto",
                color_continuous_scale="YlOrRd",
                labels=dict(x="Hour of Day (0–23)", y="Day of Week", color="Avg kW"),
            )
            fig_hm.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#94a3b8",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Insufficient multi-day telemetry for operating matrix heatmap.")

    with col_peer:
        st.subheader("👥 Peer Comparison (Same Type & Building)")
        peer_df = data.get("peer_stats")
        if peer_df is not None and not peer_df.empty and len(peer_df) > 1:
            fig_peer = px.bar(
                peer_df,
                x="asset_name",
                y="avg_power_kw",
                color="is_selected",
                color_discrete_map={True: COLORS["danger"], False: COLORS["primary"]},
                labels={"asset_name": "Equipment", "avg_power_kw": "Average Load (kW)"},
            )
            fig_peer.update_layout(
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font_color="#94a3b8",
                showlegend=False,
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_peer, use_container_width=True)
        else:
            st.info(f"No sibling equipment of type '{row.get('asset_type')}' in {row.get('building_id')} for peer benchmarking.")

    st.divider()

    # Asset Operational History: Tabular Audit Trail
    st.subheader("📋 Asset Operational History & Maintenance Audit Trail")
    history = data.get("audit_history", [])
    if not history:
        st.markdown(
            """
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid #334155; border-radius: 8px; padding: 18px; text-align: center; color: #94a3b8;">
                🟢 <b>No historical anomalies or maintenance interventions on record for this asset.</b>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Operating within calibrated nominal baseline parameters.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # History table
        hist_df = pd.DataFrame(history)
        th1, th2, th3, th4, th5, th6, th7 = st.columns([1.5, 1.4, 1.4, 3.5, 1.2, 1.2, 2.0])
        with th1:
            st.caption("**DATE / TIME**")
        with th2:
            st.caption("**TYPE**")
        with th3:
            st.caption("**IDENTIFIER**")
        with th4:
            st.caption("**EVENT / INTERVENTION**")
        with th5:
            st.caption("**SEVERITY**")
        with th6:
            st.caption("**STATUS**")
        with th7:
            st.caption("**SLA / RESOLUTION**")

        st.markdown("<hr style='margin: 4px 0 8px 0; border-color: #334155;'/>", unsafe_allow_html=True)

        for it in history:
            t1, t2, t3, t4, t5, t6, t7 = st.columns([1.5, 1.4, 1.4, 3.5, 1.2, 1.2, 2.0])
            with t1:
                st.markdown(f"<span style='font-size: 0.80rem; color: #cbd5e1;'>{it['timestamp'][:16]}</span>", unsafe_allow_html=True)
            with t2:
                badge_bg = "#0284c7" if it["record_type"] == "Work Order" else "#b91c1c"
                st.markdown(
                    f"<span style='background: {badge_bg}; color: #ffffff; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem; font-weight: 600;'>{it['record_type']}</span>",
                    unsafe_allow_html=True,
                )
            with t3:
                st.markdown(f"<code style='color: #38bdf8; font-size: 0.80rem;'>{it['id']}</code>", unsafe_allow_html=True)
            with t4:
                st.markdown(
                    f"""
                    <div style="line-height: 1.25;">
                        <span style="font-size: 0.85rem; font-weight: 600; color: #f1f5f9;">{html.escape(it['title'])}</span>
                        <div style="font-size: 0.74rem; color: #94a3b8;">{html.escape(it['details'])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with t5:
                st.markdown(severity_chip(it["severity"]), unsafe_allow_html=True)
            with t6:
                st.markdown(status_chip(it["status"]), unsafe_allow_html=True)
            with t7:
                st.markdown(f"<span style='font-size: 0.80rem; color: #f59e0b; font-weight: 500;'>{html.escape(it['sla_impact'])}</span>", unsafe_allow_html=True)

            st.markdown("<hr style='margin: 6px 0; border-color: #1e293b;'/>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Primary Destination Entry Point (render_assets)
# ─────────────────────────────────────────────────────────────────────────────

def render_assets(
    facility_meta: Any,
    assets: pd.DataFrame,
    tel: pd.DataFrame,
    ctx: pd.DataFrame,
    events: pd.DataFrame,
    complaints: Optional[pd.DataFrame] = None,
    ledger: Optional[Any] = None,
    tariff: Optional[Any] = None,
    current_user: Optional[Any] = None,
    quality_report: Optional[Any] = None,
    **kwargs: Any,
) -> None:
    """Consolidated Assets Destination View for Nivara (Step UX-6).

    Coordinates:
    - Route inspection: Check ?page=assets&asset=ASSET_ID or active_asset_detail_id.
      If asset ID specified, render Asset Detail Page.
    - Otherwise, render Primary Assets view with List View (default) vs Map View toggle.
    """
    # 1. Check deep-link query parameter or session state
    query_asset = st.query_params.get("asset")
    active_asset = query_asset or st.session_state.get("active_asset_detail_id")

    # If active asset is requested, verify it exists or fallback gracefully
    if active_asset:
        st.session_state["active_asset_detail_id"] = active_asset
        render_asset_detail_page(
            asset_id=active_asset,
            facility_meta=facility_meta,
            assets_df=assets,
            tel_df=tel,
            ctx_df=ctx,
            events_df=events,
            complaints_df=complaints,
            ledger=ledger,
        )
        return

    # 2. Main Assets Destination Header
    f_name = facility_meta.get("name") if isinstance(facility_meta, dict) else getattr(facility_meta, "facility_name", "Campus Facility")
    page_heading(
        title="Asset Registry & Intelligence",
        subtitle=f"Unified asset monitoring, telemetry health, and spatial digital twin · Facility: {f_name}",
        badge="Asset Inventory",
    )

    # 3. View Mode Toggle: 📋 List View (Default) vs 🗺️ Map View
    view_col1, view_col2 = st.columns([2, 5])
    with view_col1:
        current_view_mode = st.session_state.get("assets_main_view_mode", "📋 List View")
        selected_mode = st.radio(
            "Assets View Mode",
            ["📋 List View", "🗺️ Map View"],
            horizontal=True,
            index=0 if current_view_mode == "📋 List View" else 1,
            key="assets_main_view_mode_radio",
            label_visibility="collapsed",
        )
        st.session_state["assets_main_view_mode"] = selected_mode

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # 4. Render selected subview
    if selected_mode == "📋 List View":
        prepared_assets = prepare_asset_list_data(
            facility_meta=facility_meta,
            assets_df=assets,
            tel_df=tel,
            events_df=events,
        )
        render_asset_list_view(prepared_assets, facility_meta)
    else:
        render_asset_map_view(
            facility_meta=facility_meta,
            assets_df=assets,
            tel_df=tel,
            ctx_df=ctx,
            events_df=events,
        )
