"""Ask Nivara — Persistent, Context-Aware Slide-Over Assistant Panel.

Step UX-9 Implementation:
Converts the AI Facility Assistant from a disconnected full page into a
persistent, context-aware slide-over modal drawer accessible from any screen
via the Top Bar.

Features:
- Dynamic context detection (Asset Detail, Alert Detail, or Role/View context).
- Contextual chip & tailored quick prompts.
- Interactive conversational chat with Gemini and deterministic offline fallback.
- Guarded Action Queue with Service-Impact Guard verification (service_safe_chip)
  and 1-click dispatch to Administrator Decisions Queue or Operations Work Orders.
"""
from __future__ import annotations

import html
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

from campus_energy import assumptions
from campus_energy.decision.guard import check_service_guard
from campus_energy.decision.ledger import SavingsLedger, WorkOrderItem
from campus_energy.decision.recommendation import GuardResult, Recommendation
from campus_energy.dashboard.agent import (
    actions_to_recommendations,
    chat_with_gemini,
    generate_offline_fallback,
    parse_actions_from_response,
)
from campus_energy.profiles import FacilityProfile, load_facility_profile
from campus_energy.ui.components import service_safe_chip

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Context-Aware Detection Engine
# ─────────────────────────────────────────────────────────────────────────────

def resolve_ask_nivara_context(
    facility_meta: Optional[Any] = None,
    assets_df: Optional[pd.DataFrame] = None,
    tel_df: Optional[pd.DataFrame] = None,
    events_df: Optional[Union[pd.DataFrame, List[Any]]] = None,
    current_user: Optional[Any] = None,
    complaints_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Inspect session state, query params, and telemetry to infer active operational context.

    Detects:
    1. Asset Context: When an asset is selected or an asset detail page is active.
    2. Alert Context: When an alert detail drawer is open or alert_id is active.
    3. Role/View Context: When on Home, Energy, Water, Reports, or Settings views.

    Returns:
        Structured context dictionary containing context_type, chip_label,
        prompt_context, and detailed attributes for LLM grounding.
    """
    # ── Check 1: Alert Detail Drawer / Active Alert ──
    qp_alert = st.query_params.get("alert_id") or st.query_params.get("alert")
    state_alert = (
        st.session_state.get("active_alert_drawer_id")
        or st.session_state.get("selected_alert_id")
        or st.session_state.get("alert_id")
    )
    active_alert_id = qp_alert or state_alert

    if active_alert_id:
        alert_id_str = str(active_alert_id).strip()
        anomaly_type = "Operational Anomaly"
        asset_id = "Campus-wide"
        excess_kw = 24.5
        attribution = "Triple-engine consensus: Off-hours schedule override"

        # Search in cached active alerts list if available
        cached_alerts = st.session_state.get("cached_active_alerts", [])
        matched_alert = None
        for a in cached_alerts:
            aid = getattr(a, "alert_id", "") or str(getattr(a, "id", ""))
            if aid == alert_id_str or alert_id_str in aid:
                matched_alert = a
                break

        if matched_alert is not None:
            anomaly_type = getattr(matched_alert, "anomaly_type", anomaly_type)
            asset_id = getattr(matched_alert, "asset_id", asset_id)
            dev = getattr(matched_alert, "deviation", None)
            if dev is not None:
                excess_kw = round(float(dev), 1)
            ev = getattr(matched_alert, "evidence", "")
            if ev:
                attribution = ev
        elif events_df is not None:
            if isinstance(events_df, pd.DataFrame) and not events_df.empty:
                m_row = None
                for c in ["alert_id", "id", "event_id"]:
                    if c in events_df.columns:
                        m = events_df[events_df[c].astype(str) == alert_id_str]
                        if not m.empty:
                            m_row = m
                            break
                        m_sub = events_df[events_df[c].astype(str).str.contains(alert_id_str, na=False)]
                        if not m_sub.empty:
                            m_row = m_sub
                            break
                if m_row is not None and not m_row.empty:
                    r = m_row.iloc[0]
                    anomaly_type = str(r.get("event_type", r.get("anomaly_type", anomaly_type)))
                    asset_id = str(r.get("asset_id", asset_id))
                    excess_kw = float(r.get("excess_kw", r.get("deviation", excess_kw)))
                    attribution = str(r.get("description", attribution))

        # Check protected status for target asset
        is_protected = False
        protected_reason = ""
        if facility_meta and hasattr(facility_meta, "is_asset_protected"):
            is_protected = bool(facility_meta.is_asset_protected(asset_id))
            if is_protected and hasattr(facility_meta, "get_protected_load"):
                pl = facility_meta.get_protected_load(asset_id)
                if pl:
                    protected_reason = pl.reason or pl.category

        guard_status_str = "Blocked (Mission-Critical Protected Load)" if is_protected else "Pass (Curtailment Safe)"
        chip_label = f"🎯 Context: Alert {alert_id_str} ({anomaly_type})"
        prompt_text = (
            f"Active Operational Screen: Alert Detail Drawer for Alert '{alert_id_str}'. "
            f"Anomaly Type: {anomaly_type}. Target Equipment: {asset_id}. "
            f"Excess Power Draw: {excess_kw:.1f} kW. Attribution Factor: {attribution}. "
            f"Protected Load Status: {'PROTECTED - ' + protected_reason if is_protected else 'Standard (Curtailable)'}. "
            f"Service-Impact Guard Assessment: {guard_status_str}."
        )

        return {
            "context_type": "alert",
            "alert_id": alert_id_str,
            "anomaly_type": anomaly_type,
            "asset_id": asset_id,
            "excess_kw": excess_kw,
            "factor_attribution": attribution,
            "is_protected": is_protected,
            "protected_reason": protected_reason,
            "service_guard_status": guard_status_str,
            "chip_label": chip_label,
            "prompt_context": prompt_text,
            "context_prompt_text": prompt_text,
            "details": {
                "alert_id": alert_id_str,
                "asset_id": asset_id,
                "anomaly_type": anomaly_type,
                "excess_kw": excess_kw,
                "is_protected": is_protected,
            },
        }

    # ── Check 2: Asset Detail Page / Active Asset ──
    qp_asset = st.query_params.get("asset") or st.query_params.get("asset_id")
    state_asset = (
        st.session_state.get("active_asset_detail_id")
        or st.session_state.get("selected_asset_id")
        or st.session_state.get("asset_id")
    )
    active_asset_id = qp_asset or state_asset

    if active_asset_id:
        asset_id_str = str(active_asset_id).strip()
        asset_name = asset_id_str
        equipment_type = "Equipment"
        building_id = "BLD-MAIN"
        zone_id = "—"

        if assets_df is not None and not assets_df.empty and "asset_id" in assets_df.columns:
            m_asset = assets_df[assets_df["asset_id"].astype(str) == asset_id_str]
            if not m_asset.empty:
                r = m_asset.iloc[0]
                asset_name = str(r.get("asset_name", asset_name))
                equipment_type = str(r.get("equipment_type", r.get("type", equipment_type)))
                building_id = str(r.get("building_id", building_id))
                zone_id = str(r.get("zone_id", zone_id))

        # Protected status
        is_protected = False
        protected_reason = ""
        if facility_meta and hasattr(facility_meta, "is_asset_protected"):
            is_protected = bool(facility_meta.is_asset_protected(asset_id_str))
            if is_protected and hasattr(facility_meta, "get_protected_load"):
                pl = facility_meta.get_protected_load(asset_id_str)
                if pl:
                    protected_reason = pl.reason or pl.category

        # Telemetry metrics
        current_kw = 38.5
        baseline_delta = 6.2
        if tel_df is not None and not tel_df.empty and "asset_id" in tel_df.columns:
            a_tel = tel_df[tel_df["asset_id"].astype(str) == asset_id_str]
            if not a_tel.empty:
                if "timestamp" in a_tel.columns:
                    a_tel = a_tel.sort_values("timestamp")
                latest = a_tel.iloc[-1]
                if "power_kw" in latest:
                    current_kw = round(float(latest["power_kw"]), 1)
                mean_kw = float(a_tel["power_kw"].mean()) if "power_kw" in a_tel.columns else current_kw
                base_kw = float(latest.get("baseline_kw", mean_kw))
                baseline_delta = round(current_kw - base_kw, 1)

        chip_label = f"🎯 Context: Asset {asset_id_str} ({asset_name})"
        prompt_text = (
            f"Active Operational Screen: Asset Detail Page for Asset '{asset_id_str}' ({asset_name}). "
            f"Equipment Class: {equipment_type}. Facility: {building_id} (Zone: {zone_id}). "
            f"Current Power Draw: {current_kw:.1f} kW. Calibrated Baseline Delta: {baseline_delta:+.1f} kW. "
            f"Mission-Critical Protection: {'PROTECTED - ' + protected_reason if is_protected else 'Standard (Curtailable)'}."
        )

        return {
            "context_type": "asset",
            "asset_id": asset_id_str,
            "asset_name": asset_name,
            "equipment_type": equipment_type,
            "building_id": building_id,
            "zone_id": zone_id,
            "current_kw": current_kw,
            "baseline_delta": baseline_delta,
            "is_protected": is_protected,
            "protected_reason": protected_reason,
            "chip_label": chip_label,
            "prompt_context": prompt_text,
            "context_prompt_text": prompt_text,
            "details": {
                "asset_id": asset_id_str,
                "building_id": building_id,
                "zone_id": zone_id,
                "current_kw": current_kw,
                "is_protected": is_protected,
            },
        }

    # ── Check 3: Active View & User Persona Context ──
    cur_page = (
        st.query_params.get("page")
        or st.session_state.get("selected_nav_destination")
        or st.session_state.get("selected_nav_page")
        or "home"
    )
    role_key = (
        current_user.role
        if current_user and hasattr(current_user, "role")
        else st.session_state.get("selected_role", "administrator")
    )
    role_display = (
        current_user.role_display
        if current_user and hasattr(current_user, "role_display")
        else role_key.capitalize()
    )
    cur_page_lower = str(cur_page).lower()

    if "energy" in cur_page_lower:
        chip_label = "🎯 Context: Energy Monitoring & Peak Demand"
        prompt_text = (
            f"Active Screen: Energy Monitoring. Role: {role_display}. "
            f"Focus: Live load curve, peak demand charges (18:00–22:00 IST), ToD tariff optimization."
        )
    elif "water" in cur_page_lower:
        chip_label = "🎯 Context: Water Balance & Leak Watch"
        prompt_text = (
            f"Active Screen: Water Resources. Role: {role_display}. "
            f"Focus: Municipal intake vs borewell extraction balance, distribution pipe leaks, pumping kWh."
        )
    elif "report" in cur_page_lower:
        chip_label = "🎯 Context: Statutory Reports & Compliance"
        prompt_text = (
            f"Active Screen: Statutory Reports. Role: {role_display}. "
            f"Focus: BEE Star Rating (1–5 Stars), ECBC 2017 chapter audit, NAPCC 20% carbon burndown."
        )
    elif "setting" in cur_page_lower:
        chip_label = "🎯 Context: Platform Settings & Assumptions"
        prompt_text = (
            f"Active Screen: Platform Administration. Role: {role_display}. "
            f"Focus: 49 governed assumptions, telemetry stream health, multi-channel alerts, user directory."
        )
    elif role_key == "administrator":
        chip_label = "🎯 Context: Administrator (Executive & Budget Risk)"
        prompt_text = (
            f"Active Screen: Executive Home. Role: Administrator. "
            f"Focus: Peak demand financial surcharges, budget variance, decisions waiting executive sign-off, IPMVP ROI."
        )
    elif role_key == "operations":
        chip_label = "🎯 Context: Operations (Alerts & Work Orders)"
        prompt_text = (
            f"Active Screen: Operations Home. Role: Operations Lead. "
            f"Focus: Active equipment anomalies, work orders nearing SLA, root-cause diagnostics, Service-Impact Guard."
        )
    elif role_key == "sustainability":
        chip_label = "🎯 Context: Sustainability (ESG & Carbon Targets)"
        prompt_text = (
            f"Active Screen: Sustainability Home. Role: Sustainability Officer. "
            f"Focus: Verified Scope 2 GHG abatement, BEE Star EPI benchmarks, rooftop solar PV/BESS dispatch."
        )
    else:
        chip_label = f"🎯 Context: Facility Intelligence ({role_display})"
        prompt_text = (
            f"Active Screen: Facility Intelligence Overview. Role: {role_display}. "
            f"Facility: {getattr(facility_meta, 'facility_name', 'Campus')}."
        )

    return {
        "context_type": "role_view",
        "view": cur_page_lower,
        "role": role_key,
        "role_display": role_display,
        "chip_label": chip_label,
        "prompt_context": prompt_text,
        "context_prompt_text": prompt_text,
        "details": {
            "role": role_key,
            "view": cur_page_lower,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Contextual Quick Prompts Engine
# ─────────────────────────────────────────────────────────────────────────────

def get_contextual_quick_prompts(
    context_info: Dict[str, Any],
    user_role: str = "administrator",
) -> List[str]:
    """Generate dynamic suggested questions tailored to the active screen and role."""
    ctype = context_info.get("context_type", "role_view")

    # Asset context questions
    if ctype == "asset":
        return [
            "Explain recent power fluctuations for this asset.",
            "Is this asset protected from curtailment?",
            "Compare with peer units.",
            "What is the recommended maintenance schedule?",
        ]

    # Alert context questions
    if ctype == "alert":
        return [
            "Can we shut off this asset safely right now?",
            "What is the root cause of this alert?",
            "What is the financial cost if this alert persists for 7 days?",
            "Recommend corrective work order steps.",
        ]

    # General / Role context questions
    norm_role = str(user_role).lower().strip()
    if norm_role == "administrator":
        return [
            "What is our peak demand charge risk this month?",
            "Summarize campus operating costs and peak tariff risk.",
            "Which department has highest cost allocation variance?",
            "What is total verified financial ROI in ledger?",
        ]
    elif norm_role == "operations":
        return [
            "Which work orders are breaching SLA?",
            "What are the most critical equipment anomalies right now?",
            "Explain root cause of latest HVAC anomaly.",
            "Show me recommendations blocked by the Service Guard.",
        ]
    elif norm_role == "sustainability":
        return [
            "How can we improve our BEE star rating?",
            "How are we tracking against NAPCC 20% carbon budget?",
            "Summarize cumulative verified Scope 2 GHG abatement.",
            "What is solar PV utilization and BESS dispatch impact?",
        ]
    else:
        return [
            "What are the most critical energy anomalies right now?",
            "Which department is wasting the most energy?",
            "How can we reduce peak demand tomorrow?",
            "Summarise the last 24 hours of campus energy performance.",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Guarded Actions Queue & Verification
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_actions_with_guard(
    actions: List[Dict[str, Any]],
    facility_meta: Any,
    context: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Recommendation, GuardResult]]:
    """Convert parsed actions into Recommendation instances and evaluate via Service-Impact Guard."""
    if not actions:
        return []
    recs = actions_to_recommendations(actions, facility_meta, context=context)
    results: List[Tuple[Recommendation, GuardResult]] = []
    for r in recs:
        rec_ctx = dict(context or {})
        rec_ctx["target"] = r.target
        guard_res = check_service_guard(r, facility_meta, context=rec_ctx)
        r.service_guard = guard_res
        results.append((r, guard_res))
    return results


def _render_guarded_actions_section(
    actions: List[Dict[str, Any]],
    facility_meta: Any,
    ledger: Optional[SavingsLedger],
    current_user: Optional[Any],
    context_info: Dict[str, Any],
) -> None:
    """Render structured actions queue with Service-Impact Guard chips and dispatch triggers."""
    if not actions:
        return

    st.markdown("---")
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">
                📋 Guarded Action Queue
            </span>
            <span style="font-size: 0.78rem; color: #94a3b8;">
                Verified by Service-Impact Guard
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evaluated = evaluate_actions_with_guard(actions, facility_meta, context=context_info.get("details"))
    role_key = current_user.role if current_user and hasattr(current_user, "role") else "administrator"

    for idx, (rec, guard_res) in enumerate(evaluated):
        badge_html = service_safe_chip(
            status=guard_res.status,
            reason=guard_res.reason,
            safe_alternative=guard_res.safe_alternative,
        )

        with st.container():
            c_info, c_chip = st.columns([5, 3])
            with c_info:
                st.markdown(
                    f"**{rec.action_text}**  \n"
                    f"<span style='color: #94a3b8; font-size: 0.82rem;'>"
                    f"Target: <code>{html.escape(rec.target)}</code> · "
                    f"Estimated Saving: <b>{rec.savings.kwh.expected:.1f} kWh</b> "
                    f"[{rec.savings.kwh.low:.1f} – {rec.savings.kwh.high:.1f}] · "
                    f"Avoided: <b>₹{rec.savings.inr.expected:,.0f}</b>"
                    f"</span>",
                    unsafe_allow_html=True,
                )
            with c_chip:
                st.markdown(badge_html, unsafe_allow_html=True)

            # Blocked state warning and safe alternative dispatch
            if guard_res.status == "blocked":
                st.markdown(
                    f"""
                    <div style="background: rgba(244, 63, 94, 0.12); border-left: 3px solid #f43f5e; padding: 6px 10px; border-radius: 4px; font-size: 0.8rem; color: #fecdd3; margin-top: 4px; margin-bottom: 6px;">
                        <b>🛡️ Blocked Reason:</b> {html.escape(guard_res.reason or 'Mission-critical load')}
                        {f'<br/><b>Safe Alternative:</b> {html.escape(guard_res.safe_alternative)}' if guard_res.safe_alternative else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if guard_res.safe_alternative and ledger is not None:
                    if st.button(
                        f"🔧 Dispatch Safe Alternative: {guard_res.safe_alternative[:28]}…",
                        key=f"btn_alt_dispatch_{rec.id}_{idx}",
                        use_container_width=True,
                    ):
                        alt_rec = Recommendation(
                            id=f"ALT-{rec.id}",
                            facility_id=rec.facility_id,
                            type="sensor_check",
                            target=rec.target,
                            action_text=f"Safe Alternative: {guard_res.safe_alternative} on {rec.target}",
                            trigger=rec.trigger,
                            savings=rec.savings,
                            cost=0.0,
                            owner_role="operations",
                        )
                        item = ledger.add_recommendation(alt_rec)
                        wo = ledger.create_work_order(
                            item.work_order_id,
                            assignee=current_user.name if current_user else "Operations Lead",
                            sla_hours=12,
                            priority="high",
                        )
                        st.success(f"Dispatched safe alternative Work Order {wo.work_order_id}!")
                        st.rerun()

            # Permitted / Review actions: Add to Decisions or Create Work Order
            else:
                col_btn_a, col_btn_b = st.columns(2)
                with col_btn_a:
                    if st.button(
                        "📋 Add to Decisions Queue",
                        key=f"btn_add_dec_{rec.id}_{idx}",
                        use_container_width=True,
                        help="Submit to Administrator's Decisions Queue for executive review",
                    ):
                        if ledger is not None:
                            item = ledger.add_recommendation(rec)
                            st.success(f"Added to Administrator Decisions Queue (Ref: {item.work_order_id})")
                            st.rerun()
                        else:
                            st.info("Added to Decisions Queue.")
                with col_btn_b:
                    if st.button(
                        "🚀 Create Work Order",
                        key=f"btn_create_wo_{rec.id}_{idx}",
                        use_container_width=True,
                        help="Dispatch as an active field work order with 24h SLA",
                    ):
                        if ledger is not None:
                            item = ledger.add_recommendation(rec)
                            wo = ledger.create_work_order(
                                item.work_order_id,
                                assignee=current_user.name if current_user else "Operations Lead",
                                sla_hours=24,
                                priority="high",
                            )
                            st.success(f"Work Order {wo.work_order_id} created and dispatched!")
                            st.rerun()
                        else:
                            st.info("Work Order created and dispatched.")

            st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px dashed #334155;'/>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Proposed Actions", key="btn_clear_ask_nivara_actions", use_container_width=False):
        st.session_state["ask_nivara_actions"] = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Interactive Slide-Over Panel Content
# ─────────────────────────────────────────────────────────────────────────────

def _render_ask_nivara_content(
    current_user: Optional[Any],
    facility_meta: Any,
    assets: Optional[pd.DataFrame],
    tel: Optional[pd.DataFrame],
    ctx: Optional[pd.DataFrame],
    events: Optional[Union[pd.DataFrame, List[Any]]],
    complaints: Optional[pd.DataFrame],
    ledger: Optional[SavingsLedger],
    tariff: Optional[Any],
    api_key: Optional[str] = None,
) -> None:
    """Render full content stream of Ask Nivara modal drawer."""
    # Resolve dynamic active context
    context_info = resolve_ask_nivara_context(
        facility_meta=facility_meta,
        assets_df=assets,
        tel_df=tel,
        events_df=events,
        current_user=current_user,
        complaints_df=complaints,
    )

    role_key = current_user.role if current_user and hasattr(current_user, "role") else "administrator"
    role_display = current_user.role_display if current_user and hasattr(current_user, "role_display") else role_key.capitalize()
    fac_name = getattr(facility_meta, "facility_name", "Campus Facility")

    # ── Top Bar / Header of Drawer ──
    col_hdr, col_cls = st.columns([6, 1])
    with col_hdr:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">
                    ✨ Ask Nivara
                </span>
                <span class="chip" style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; font-weight: 600; font-size: 0.78rem; padding: 2px 8px; border-radius: 4px;">
                    {html.escape(context_info['chip_label'])}
                </span>
            </div>
            <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 6px;">
                Role Mandate: <b>{html.escape(role_display)}</b> · Facility: <b>{html.escape(fac_name)}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_cls:
        if st.button("✖ Close", key="btn_close_ask_nivara_panel", help="Close slide-over assistant", use_container_width=True):
            st.session_state["show_ask_nivara"] = False
            st.rerun()

    # ── Optional Model / Key Settings (Discreet) ──
    with st.expander("⚙️ Assistant Settings & API Configuration", expanded=False):
        key_input = st.text_input(
            "Gemini API Key (Optional)",
            value=api_key or st.session_state.get("gemini_api_key", ""),
            type="password",
            help="Leave blank to run 100% locally with rule-based grounded intelligence, or enter a Gemini key from https://aistudio.google.com/app/apikey.",
            key="ask_nivara_api_key_input",
        )
        if key_input:
            st.session_state["gemini_api_key"] = key_input.strip()
            api_key = key_input.strip()
        else:
            st.caption("ℹ️ Operating in Grounded Offline Facility Intelligence Mode. Generates deterministic actions calibrated against live telemetry, active alerts, and the assumptions registry.")

    # ── Contextual Quick Prompts ──
    quick_prompts = get_contextual_quick_prompts(context_info, user_role=role_key)
    st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #cbd5e1; margin-top: 4px; margin-bottom: 4px;'>Suggested Questions:</div>", unsafe_allow_html=True)
    qp_cols = st.columns(len(quick_prompts))
    for i, qp in enumerate(quick_prompts):
        with qp_cols[i]:
            btn_label = qp if len(qp) <= 32 else qp[:30] + "…"
            if st.button(btn_label, key=f"btn_ask_nivara_qp_{i}", help=qp, use_container_width=True):
                st.session_state["ask_nivara_pending_prompt"] = qp
                st.rerun()

    st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid #334155;'/>", unsafe_allow_html=True)

    # ── Chat Messages Stream ──
    if "ask_nivara_messages" not in st.session_state or not st.session_state["ask_nivara_messages"]:
        greeting = (
            f"Hello {current_user.name if current_user else 'Facility Operator'}! "
            f"I am Nivara, your facility intelligence assistant calibrated for {role_display} operations. "
            f"I am currently grounded in the context of **{context_info['chip_label'].replace('🎯 Context: ', '')}**. "
            f"How can I assist you with {fac_name} today?"
        )
        st.session_state["ask_nivara_messages"] = [{"role": "assistant", "content": greeting}]

    chat_box = st.container()
    with chat_box:
        for msg in st.session_state["ask_nivara_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Guarded Action Queue ──
    if "ask_nivara_actions" not in st.session_state:
        st.session_state["ask_nivara_actions"] = []

    _render_guarded_actions_section(
        actions=st.session_state["ask_nivara_actions"],
        facility_meta=facility_meta,
        ledger=ledger,
        current_user=current_user,
        context_info=context_info,
    )

    # ── Chat Input & Processing ──
    user_input = st.chat_input(
        f"Ask Nivara about {context_info['chip_label'].replace('🎯 Context: ', '')}…",
        key="ask_nivara_chat_input",
    )

    # Handle quick prompt selection
    if "ask_nivara_pending_prompt" in st.session_state:
        user_input = st.session_state.pop("ask_nivara_pending_prompt")

    if user_input:
        st.session_state["ask_nivara_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing operational telemetry…"):
                active_api_key = api_key or st.session_state.get("gemini_api_key")
                raw_events = events if isinstance(events, pd.DataFrame) else pd.DataFrame()
                raw_assets = assets if isinstance(assets, pd.DataFrame) else pd.DataFrame()
                raw_complaints = complaints if isinstance(complaints, pd.DataFrame) else pd.DataFrame()

                # Dispatch to LLM or deterministic grounded fallback with context_data
                response = chat_with_gemini(
                    api_key=active_api_key,
                    messages=st.session_state["ask_nivara_messages"],
                    events_df=raw_events,
                    assets_df=raw_assets,
                    complaints_df=raw_complaints,
                    active_role=role_key,
                    profile=facility_meta,
                    context_data=context_info,
                )
                st.markdown(response)

        st.session_state["ask_nivara_messages"].append({"role": "assistant", "content": response})

        # Parse and queue structured actions
        new_actions = parse_actions_from_response(response)
        if new_actions:
            existing_types = {a.get("action_type") for a in st.session_state["ask_nivara_actions"]}
            for act in new_actions:
                if act.get("action_type") not in existing_types:
                    st.session_state["ask_nivara_actions"].append(act)
                    existing_types.add(act.get("action_type"))

        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Modal Dialog Wrapper & Panel Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if hasattr(st, "dialog"):
    @st.dialog("✨ Ask Nivara — AI Facility Assistant", width="large")
    def _render_ask_nivara_dialog_wrapper(
        current_user: Optional[Any],
        facility_meta: Any,
        assets: Optional[pd.DataFrame],
        tel: Optional[pd.DataFrame],
        ctx: Optional[pd.DataFrame],
        events: Optional[Union[pd.DataFrame, List[Any]]],
        complaints: Optional[pd.DataFrame],
        ledger: Optional[SavingsLedger],
        tariff: Optional[Any],
        api_key: Optional[str] = None,
    ) -> None:
        _render_ask_nivara_content(
            current_user=current_user,
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=ledger,
            tariff=tariff,
            api_key=api_key,
        )
else:
    def _render_ask_nivara_dialog_wrapper(
        current_user: Optional[Any],
        facility_meta: Any,
        assets: Optional[pd.DataFrame],
        tel: Optional[pd.DataFrame],
        ctx: Optional[pd.DataFrame],
        events: Optional[Union[pd.DataFrame, List[Any]]],
        complaints: Optional[pd.DataFrame],
        ledger: Optional[SavingsLedger],
        tariff: Optional[Any],
        api_key: Optional[str] = None,
    ) -> None:
        with st.expander("✨ Ask Nivara — AI Facility Assistant", expanded=True):
            _render_ask_nivara_content(
                current_user=current_user,
                facility_meta=facility_meta,
                assets=assets,
                tel=tel,
                ctx=ctx,
                events=events,
                complaints=complaints,
                ledger=ledger,
                tariff=tariff,
                api_key=api_key,
            )


def render_ask_nivara_panel(
    current_user: Optional[Any] = None,
    facility_meta: Optional[Any] = None,
    assets: Optional[pd.DataFrame] = None,
    tel: Optional[pd.DataFrame] = None,
    ctx: Optional[pd.DataFrame] = None,
    events: Optional[Union[pd.DataFrame, List[Any]]] = None,
    complaints: Optional[pd.DataFrame] = None,
    ledger: Optional[SavingsLedger] = None,
    tariff: Optional[Any] = None,
    api_key: Optional[str] = None,
    force_open: bool = False,
    **kwargs: Any,
) -> None:
    """Render the persistent context-aware Ask Nivara slide-over panel.

    Args:
        current_user: Active User instance.
        facility_meta: Active FacilityProfile.
        assets: Assets DataFrame.
        tel: Telemetry DataFrame.
        ctx: Context/weather DataFrame.
        events: Events or alerts collection.
        complaints: Occupant complaints DataFrame.
        ledger: Active SavingsLedger instance.
        tariff: Active IndianTariff instance.
        api_key: Optional Gemini API key string.
        force_open: When True, renders drawer even if session_state flag is False.
    """
    if not (st.session_state.get("show_ask_nivara", False) or force_open):
        return

    # Fallbacks for standalone testing or uninitialized callers
    if facility_meta is None:
        try:
            facility_meta = load_facility_profile("config/facility_profiles/university.yaml")
        except Exception:
            facility_meta = None

    if ledger is None:
        ledger = SavingsLedger()

    try:
        _render_ask_nivara_dialog_wrapper(
            current_user=current_user,
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=ledger,
            tariff=tariff,
            api_key=api_key,
        )
    except Exception as e:
        logger.debug("Dialog execution fallback to direct content render: %s", e)
        _render_ask_nivara_content(
            current_user=current_user,
            facility_meta=facility_meta,
            assets=assets,
            tel=tel,
            ctx=ctx,
            events=events,
            complaints=complaints,
            ledger=ledger,
            tariff=tariff,
            api_key=api_key,
        )
