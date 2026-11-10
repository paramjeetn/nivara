"""Nivara Authentication & Login Stub Screen.

Provides a clean, professional sign-in interface offering one-click demo profiles
for each operational persona or direct institutional email/ID authentication.
"""
from __future__ import annotations

import logging
from typing import Optional
import streamlit as st

from campus_energy.users import UserProfile, authenticate_user, get_demo_users

logger = logging.getLogger(__name__)


def render_login_screen() -> Optional[UserProfile]:
    """Render the clean authentication screen and return authenticated UserProfile if signed in."""

    st.markdown(
        """
        <style>
            .login-hero {
                text-align: center;
                margin-top: 40px;
                margin-bottom: 30px;
            }
            .login-brand-title {
                font-size: 2.6rem;
                font-weight: 800;
                background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 6px;
            }
            .login-subtitle {
                font-size: 1.05rem;
                color: #94a3b8;
                font-weight: 400;
                max-width: 580px;
                margin: 0 auto;
            }
            .profile-card {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 16px;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                transition: border-color 0.2s ease;
            }
            .profile-card:hover {
                border-color: #6366f1;
            }
            .profile-role-tag {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-bottom: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hero Banner
    st.markdown(
        """
        <div class="login-hero">
            <div class="login-brand-title">⚡ Nivara Enterprise</div>
            <div class="login-subtitle">
                Operational Campus Energy, IoT Telemetry & Facility Intelligence Platform
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-top: 8px;">
                Biju Patnaik University of Technology (BPUT) Central Management System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Center card container
    _, main_col, _ = st.columns([1, 6, 1])

    with main_col:
        st.markdown(
            """
            <div style="background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 24px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);">
                <div style="margin-bottom: 18px;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">Select Operational Persona</span>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">
                        Sign in with one of the pre-configured institutional profiles below:
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        demo_users = get_demo_users()
        user_cols = st.columns(len(demo_users))

        for idx, (col, user) in enumerate(zip(user_cols, demo_users)):
            with col:
                role_color = {
                    "administrator": "#38bdf8",
                    "operations": "#f59e0b",
                    "sustainability": "#34d399",
                }.get(user.role, "#818cf8")

                st.markdown(
                    f"""
                    <div class="profile-card">
                        <div>
                            <span class="profile-role-tag" style="background: rgba({int(role_color[1:3], 16)}, {int(role_color[3:5], 16)}, {int(role_color[5:7], 16)}, 0.15); color: {role_color}; border: 1px solid {role_color};">
                                {user.role_icon} {user.role_display}
                            </span>
                            <div style="font-weight: 700; color: #f8fafc; font-size: 0.98rem; margin-top: 4px;">
                                {user.name}
                            </div>
                            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px; line-height: 1.3;">
                                {user.title}
                            </div>
                            <div style="font-size: 0.72rem; color: #64748b; margin-top: 8px;">
                                ✉️ {user.email}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                btn_key = f"login_btn_{user.id}"
                if st.button(
                    f"Sign In as {user.role_display}",
                    key=btn_key,
                    use_container_width=True,
                    help=f"Log in as {user.name} ({user.role_display})",
                ):
                    st.session_state["user"] = user
                    st.session_state["active_persona_key"] = user.role
                    st.session_state["selected_nav_page"] = "Home"
                    st.session_state["current_facility"] = user.default_facility
                    st.session_state["time_range"] = "Today"
                    st.query_params["page"] = "home"
                    st.query_params["facility"] = user.default_facility
                    st.query_params["range"] = "today"
                    st.rerun()

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.divider()

        # Email / ID input option
        st.markdown(
            """
            <div style="margin-bottom: 8px; font-size: 0.88rem; font-weight: 600; color: #cbd5e1;">
                Or Sign In with Institutional Email or User ID
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_inp, col_btn = st.columns([4, 1.2])
        with col_inp:
            credential_input = st.text_input(
                "Email or User ID",
                placeholder="e.g. arun.patnaik@bput.ac.in, usr_ops_01, or priya.das@bput.ac.in",
                label_visibility="collapsed",
                key="manual_login_input",
            )
        with col_btn:
            submit_login = st.button("Sign In", key="manual_login_btn", use_container_width=True)

        if submit_login:
            if not credential_input.strip():
                st.warning("Please enter your email or user ID.")
            else:
                matched_user = authenticate_user(credential_input.strip())
                if matched_user:
                    st.session_state["user"] = matched_user
                    st.session_state["active_persona_key"] = matched_user.role
                    st.session_state["selected_nav_page"] = "Home"
                    st.session_state["current_facility"] = matched_user.default_facility
                    st.session_state["time_range"] = "Today"
                    st.query_params["page"] = "home"
                    st.query_params["facility"] = matched_user.default_facility
                    st.query_params["range"] = "today"
                    st.success(f"Welcome back, {matched_user.name}!")
                    st.rerun()
                else:
                    st.error("User not found. Please select one of the demo profiles above or check your ID.")

    return st.session_state.get("user")
