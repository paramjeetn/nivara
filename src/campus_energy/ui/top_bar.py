"""Nivara Top Bar & Global Header Component.

Provides the persistent header displaying:
- Left: Nivara brand and facility selector
- Center: Global time range selector (Today, 7 Days, 28 Days, Custom)
- Right: Data freshness indicator, alert bell with badge count, Ask Nivara button, and user profile menu.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union

import streamlit as st

from campus_energy.users import UserProfile

logger = logging.getLogger(__name__)

TIME_RANGE_OPTIONS = ["Today", "7 Days", "28 Days", "Custom"]


def render_top_bar(
    user: UserProfile,
    facility_options: dict[str, Any],
    current_facility_key: str,
    open_alerts_count: int = 4,
    latest_timestamp: Optional[Union[datetime, str]] = None,
) -> str:
    """Render the top bar and return the currently selected facility key."""

    # Top bar CSS styles for compact, executive styling
    st.markdown(
        """
        <style>
            .nivara-top-bar {
                background: #0f172a;
                border-bottom: 1px solid #334155;
                padding: 8px 16px;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-radius: 8px;
            }
            .freshness-badge-live {
                background: rgba(34, 197, 94, 0.12);
                border: 1px solid #22c55e;
                color: #4ade80;
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 0.78rem;
                font-weight: 600;
                display: inline-block;
                white-space: nowrap;
                text-align: center;
            }
            .freshness-badge-stale {
                background: rgba(245, 158, 11, 0.12);
                border: 1px solid #f59e0b;
                color: #fbbf24;
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 0.78rem;
                font-weight: 600;
                display: inline-block;
                white-space: nowrap;
                text-align: center;
            }
            .user-avatar-badge {
                background: #3b82f6;
                color: #ffffff;
                width: 28px;
                height: 28px;
                border-radius: 50%;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                font-weight: 700;
                font-size: 0.76rem;
                margin-right: 6px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Filter facility options according to user authorization
    allowed_facilities: dict[str, Any] = {}
    facility_key_to_label: dict[str, str] = {}
    label_to_facility_key: dict[str, str] = {}

    for label, meta in facility_options.items():
        fkey = meta["key"] if isinstance(meta, dict) else getattr(meta, "key", "")
        if not fkey:
            fkey = getattr(meta, "facility_id", "university")
        if user.has_facility_access(fkey):
            allowed_facilities[label] = meta
            facility_key_to_label[fkey] = label
            label_to_facility_key[label] = fkey

    if not allowed_facilities:
        allowed_facilities = facility_options
        for label, meta in facility_options.items():
            fkey = meta["key"] if isinstance(meta, dict) else getattr(meta, "key", "")
            facility_key_to_label[fkey] = label
            label_to_facility_key[label] = fkey

    # Determine selected label
    current_label = facility_key_to_label.get(current_facility_key, list(allowed_facilities.keys())[0])

    # Top Bar Columns
    col_brand, col_facility, col_range, col_freshness, col_alerts, col_ask, col_user = st.columns(
        [1.6, 2.2, 2.5, 1.8, 0.8, 1.2, 2.3],
        vertical_alignment="center",
    )

    selected_facility_key = current_facility_key

    # 1. Brand Logo / Mark
    with col_brand:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.35rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">⚡ Nivara</span>
                <span style="font-size: 0.70rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; border-left: 1px solid #334155; padding-left: 6px;">Ops</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Facility Selector
    with col_facility:
        labels_list = list(allowed_facilities.keys())
        idx = labels_list.index(current_label) if current_label in labels_list else 0
        if len(labels_list) > 1:
            chosen_label = st.selectbox(
                "Facility",
                labels_list,
                index=idx,
                label_visibility="collapsed",
                key="top_bar_facility_select",
                help="Switch current campus or facility view",
            )
            selected_facility_key = label_to_facility_key.get(chosen_label, current_facility_key)
            if selected_facility_key != current_facility_key:
                st.session_state["current_facility"] = selected_facility_key
                st.query_params["facility"] = selected_facility_key
                st.rerun()
        else:
            st.markdown(
                f"<div style='font-size: 0.85rem; font-weight: 600; color: #cbd5e1;'>{current_label}</div>",
                unsafe_allow_html=True,
            )

    # 3. Global Time Range Selector
    with col_range:
        current_time_range = st.session_state.get("time_range", "Today")
        if current_time_range not in TIME_RANGE_OPTIONS:
            current_time_range = "Today"

        try:
            chosen_range = st.segmented_control(
                "Time Range",
                TIME_RANGE_OPTIONS,
                default=current_time_range,
                label_visibility="collapsed",
                key="top_bar_time_range_control",
                help="Select global operational analysis time window",
            )
        except Exception:
            chosen_range = st.selectbox(
                "Time Range",
                TIME_RANGE_OPTIONS,
                index=TIME_RANGE_OPTIONS.index(current_time_range),
                label_visibility="collapsed",
                key="top_bar_time_range_control_fallback",
            )

        if chosen_range and chosen_range != current_time_range:
            st.session_state["time_range"] = chosen_range
            range_slug = chosen_range.lower().replace(" ", "_")
            st.query_params["range"] = range_slug
            st.rerun()

    # 4. Data Freshness Indicator
    with col_freshness:
        # Determine timestamp display
        ts_display = "Live"
        is_live = True
        if latest_timestamp:
            if isinstance(latest_timestamp, datetime):
                ts_display = latest_timestamp.strftime("%H:%M IST")
            elif isinstance(latest_timestamp, str):
                if "T" in latest_timestamp:
                    parts = latest_timestamp.split("T")
                    ts_display = f"{parts[1][:5]} IST" if len(parts) > 1 else latest_timestamp
                else:
                    ts_display = latest_timestamp[-8:-3] + " IST" if len(latest_timestamp) >= 8 else latest_timestamp

        badge_class = "freshness-badge-live" if is_live else "freshness-badge-stale"
        icon = "🟢" if is_live else "🟡"
        st.markdown(
            f"""
            <div class="{badge_class}" title="Telemetry feed status">
                {icon} Live: {ts_display}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Alert Bell with badge count
    with col_alerts:
        bell_label = f"🔔 {open_alerts_count}"
        if st.button(bell_label, key="top_bar_alerts_bell", help="View active alerts and open work orders", use_container_width=True):
            st.session_state["selected_nav_page"] = "Alerts & Work"
            st.query_params["page"] = "alerts"
            st.rerun()

    # 6. Ask Nivara Assistant Button
    with col_ask:
        if st.button("✨ Ask Nivara", key="top_bar_ask_btn", help="Open AI Facility Assistant drawer", use_container_width=True):
            st.session_state["show_ask_nivara"] = not st.session_state.get("show_ask_nivara", False)
            st.rerun()

    # 7. User Menu with Popover
    with col_user:
        short_name = user.name.split()[0] if not user.name.startswith("Dr.") else " ".join(user.name.split()[:2])
        menu_label = f"{user.role_icon} {short_name}"

        with st.popover(menu_label, use_container_width=True):
            st.markdown(
                f"""
                <div style="padding: 4px 2px;">
                    <div style="font-weight: 700; color: #f8fafc; font-size: 1.0rem;">
                        {user.role_icon} {user.name}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.82rem; margin-top: 2px;">
                        {user.title}
                    </div>
                    <div style="margin-top: 8px; font-size: 0.78rem; color: #cbd5e1;">
                        <span style="background: #1e293b; border: 1px solid #334155; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                            {user.role_display}
                        </span>
                        <span style="color: #64748b; margin-left: 6px;">ID: <code>{user.id}</code></span>
                    </div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 6px;">
                        ✉️ {user.email}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.divider()

            col_sw, col_so = st.columns(2)
            with col_sw:
                if st.button("🔄 Switch", key="popover_switch_user", help="Switch operational user profile", use_container_width=True):
                    st.session_state.pop("user", None)
                    st.session_state.pop("active_persona_key", None)
                    st.query_params.clear()
                    st.rerun()

            with col_so:
                if st.button("🚪 Sign Out", key="popover_sign_out", help="Sign out of Nivara session", use_container_width=True):
                    st.session_state.pop("user", None)
                    st.session_state.pop("active_persona_key", None)
                    st.query_params.clear()
                    st.rerun()

    st.markdown("<hr style='margin: 4px 0 14px 0; border: none; border-bottom: 1px solid #334155;'/>", unsafe_allow_html=True)

    return selected_facility_key
