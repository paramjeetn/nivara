"""Nivara Styleguide & Design System Showcase (`?page=styleguide`).

Living component specification and interactive preview for:
- Typography hierarchy & tabular numerals
- 12-column responsive layout grid
- Dense, scannable KPI metric cards
- Status chips & severity badges
- ServiceGuard safety check chips (pass, needs_review, blocked)
- Indian currency formatting (INR)
- Relative human timestamp formatting
- Enterprise data tables
- Clean empty states & calculation traceability popovers
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import streamlit as st

from campus_energy.ui.components import (
    empty_state,
    format_inr,
    format_inr_compact,
    format_relative_time,
    info_popover,
    kpi_card,
    service_safe_chip,
    severity_chip,
    status_chip,
)
from campus_energy.ui.styles import load_global_styles

logger = logging.getLogger(__name__)


def render_styleguide() -> None:
    """Render the comprehensive Nivara design system styleguide."""
    # Ensure styles are loaded
    load_global_styles()

    st.markdown(
        """
        <div style="display: flex; align-items: baseline; justify-content: space-between; border-bottom: 1px solid #334155; padding-bottom: 12px; margin-bottom: 20px;">
          <div>
            <h1 class="nivara-heading-1"><span class="heading-accent"></span>Nivara Design System & Component Library</h1>
            <p class="nivara-caption">Quiet, dense, high-contrast visual language for enterprise facility operations.</p>
          </div>
          <div>
            <span class="chip chip-status chip-status-verified"><span class="chip-dot"></span>Design Token Spec v2.4</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([
        "1. Typography & Colors",
        "2. KPI Metric Cards",
        "3. Badges & Chips",
        "4. Service-Safe Guard",
        "5. Number & Time Formatting",
        "6. Tables & Empty States",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1: Typography & Design Tokens
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Typography Hierarchy")
        st.caption("Clean, sans-serif type scale without raw decorative emojis in headings.")

        st.markdown(
            """
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px 24px; margin-bottom: 16px;">
              <h1 class="nivara-heading-1"><span class="heading-accent"></span>Heading 1 (1.65rem / 700) — Campus Operational Overview</h1>
              <h2 class="nivara-heading-2"><span class="heading-accent"></span>Heading 2 (1.35rem / 600) — Active Work Orders & Triaged Anomalies</h2>
              <h3 class="nivara-heading-3">Heading 3 (1.10rem / 600) — Subsystem Telemetry & Energy Disaggregation</h3>
              <p style="font-size: 0.9rem; color: #cbd5e1; margin-top: 10px; line-height: 1.5;">
                Body Text (0.90rem / 400) — Nivara monitors 142 discrete electrical meters across 6 institutional facility zones. 
                All measurements are validated against historical baselines and environmental conditions.
              </p>
              <span class="nivara-caption">Caption Text (0.80rem / 400) — Telemetry source: Odisha Discom TPCODL / IPMVP Option B Verified.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Tabular Numerals Comparison")
        st.caption("Notice how numbers with tabular figures maintain fixed character widths for perfect column scanning.")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.markdown(
                """
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px;">
                  <b style="color: #38bdf8; font-size: 0.85rem;">Tabular Numbers (<code>font-variant-numeric: tabular-nums</code>)</b>
                  <div class="tabular-nums" style="font-size: 1.15rem; color: #f8fafc; font-family: sans-serif; line-height: 1.7; margin-top: 8px;">
                    111,111.11 kW<br/>
                    888,888.88 kW<br/>
                    127,500.00 INR<br/>
                    902,341.00 INR
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_t2:
            st.markdown(
                """
                <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px;">
                  <b style="color: #94a3b8; font-size: 0.85rem;">Proportional Numbers (Standard)</b>
                  <div style="font-size: 1.15rem; color: #cbd5e1; font-family: sans-serif; line-height: 1.7; margin-top: 8px;">
                    111,111.11 kW<br/>
                    888,888.88 kW<br/>
                    127,500.00 INR<br/>
                    902,341.00 INR
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Color Tokens")
        colors = [
            ("Canvas (slate-900)", "#0f172a", "#f8fafc"),
            ("Surface (slate-800)", "#1e293b", "#f8fafc"),
            ("Border (slate-700)", "#334155", "#f8fafc"),
            ("Primary Brand (sky-600)", "#0284c7", "#ffffff"),
            ("Success (emerald-500)", "#10b981", "#ffffff"),
            ("Warning (amber-500)", "#f59e0b", "#000000"),
            ("Danger (red-500)", "#ef4444", "#ffffff"),
            ("Indigo (verified)", "#6366f1", "#ffffff"),
        ]
        cols = st.columns(len(colors))
        for col, (name, hex_code, text_color) in zip(cols, colors):
            with col:
                st.markdown(
                    f"""
                    <div style="background: {hex_code}; color: {text_color}; padding: 12px 8px; border-radius: 6px; border: 1px solid #334155; text-align: center; min-height: 75px;">
                      <div style="font-size: 0.72rem; font-weight: 600;">{name}</div>
                      <div style="font-size: 0.76rem; font-family: monospace; margin-top: 4px;">{hex_code}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2: Dense KPI Metric Cards
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### Dense KPI Metric Cards")
        st.caption("Scannable, compact layout with single primary number and popover calculation traceability.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card(
                label="Active Campus Load",
                value="482.4 kW",
                delta="-12.3 kW vs baseline",
                delta_color="normal",
                subtitle="Baseline: 494.7 kW",
                help_text="Real-time aggregate consumption across all active incomers.",
            )
        with c2:
            kpi_card(
                label="Avoided Surcharge",
                value="₹1,27,500",
                delta="+₹18,400 this cycle",
                delta_color="normal",
                subtitle="Peak shaving active",
                help_text="Savings accumulated from ToD peak shaving dispatch.",
                popover_title="Avoided Demand Surcharge Calculation",
                popover_calc_notes="Calculated as `max(0, recorded_peak_kva - contracted_demand_kva) * tariff_penalty_rate`. See TPCODL Tariff Schedule RST-2024.",
                popover_assumptions=["tariff.demand_charge_industrial", "tariff.electricity_commercial"],
            )
        with c3:
            kpi_card(
                label="Grid Peak Demand",
                value="612.0 kVA",
                delta="+32.0 kVA exceedance",
                delta_color="inverse",
                subtitle="Contract: 580.0 kVA",
                help_text="Recorded 15-minute maximum apparent power demand.",
            )
        with c4:
            kpi_card(
                label="GHG Scope 2 Today",
                value="3.82 tCO₂e",
                delta="-8.4% vs 28d avg",
                delta_color="normal",
                subtitle="CEA Grid Factor: 0.82",
                popover_title="Scope 2 Emissions Traceability",
                popover_calc_notes="Emissions calculated using the Central Electricity Authority (CEA) Version 19 grid emissions baseline factor.",
                popover_assumptions=["grid.emissions_factor_india_cea", "solar.specific_yield_bhubaneswar"],
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3: Badges & Chips
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### Operational Lifecycle Status Chips")
        st.caption("Semantic status chips with subtle backgrounds and high-contrast text.")

        statuses = ["active", "in_progress", "done", "verified", "proposed", "snoozed", "closed"]
        status_html = " ".join([status_chip(s) for s in statuses])
        st.markdown(f'<div style="padding: 12px 0;">{status_html}</div>', unsafe_allow_html=True)

        st.markdown("### Severity Chips")
        st.caption("Semantic alert and anomaly severity indicators.")

        severities = ["critical", "high", "medium", "low"]
        severity_html = " ".join([severity_chip(s) for s in severities])
        st.markdown(f'<div style="padding: 12px 0;">{severity_html}</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 4: Service-Safe Recommendation Chips
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### ServiceGuard Recommendation Safety Chips")
        st.caption("Hover over chips with warnings or blocks to inspect the documented operational reason and safe alternative.")

        safe_demo_html = f"""
        <div style="background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 18px; margin-bottom: 16px;">
          <div style="display: flex; gap: 16px; align-items: center; margin-bottom: 14px;">
            {service_safe_chip('pass')}
            <span style="font-size: 0.85rem; color: #cbd5e1;"><b>Safe to Act:</b> Automated setpoint adjustment or shed satisfies all comfort & safety constraints.</span>
          </div>
          <div style="display: flex; gap: 16px; align-items: center; margin-bottom: 14px;">
            {service_safe_chip('needs_review', reason='Zone occupancy is at 88%; cooling reduction may violate PMV < 0.70 comfort bounds.')}
            <span style="font-size: 0.85rem; color: #cbd5e1;"><b>Needs Review:</b> Requires supervisory confirmation before dispatching control command.</span>
          </div>
          <div style="display: flex; gap: 16px; align-items: center;">
            {service_safe_chip('blocked', reason='Chiller CH-1 is currently supplying critical ICU / Server Room loads.', safe_alternative='Pre-cool thermal buffer tank or throttle Administrative Block AHU-3.')}
            <span style="font-size: 0.85rem; color: #cbd5e1;"><b>Not Recommended (Blocked):</b> Action blocked by critical protected-load constraints with recommended alternative.</span>
          </div>
        </div>
        """
        st.markdown(safe_demo_html, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 5: Number & Time Formatting
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[4]:
        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("### Indian Currency Formatting (`format_inr`)")
            st.caption("Conforms to Indian grouping notation (Lakhs & Crores).")

            test_amounts = [
                (750, 0, False),
                (127500, 0, False),
                (1250000, 0, False),
                (1250000, 1, True),
                (14000000, 1, True),
                (-25000, 0, False),
                (54200.75, 2, False),
            ]

            currency_rows = []
            for amt, prec, compact in test_amounts:
                currency_rows.append({
                    "Raw Value": str(amt),
                    "Compact": str(compact),
                    "Precision": str(prec),
                    "Formatted INR": format_inr(amt, precision=prec, compact=compact),
                })
            st.dataframe(pd.DataFrame(currency_rows), use_container_width=True, hide_index=True)

        with col_f2:
            st.markdown("### Relative Time Formatting (`format_relative_time`)")
            st.caption("Human-readable operational time with clean relative descriptions.")

            now = datetime.now()
            time_samples = [
                ("15 seconds ago", now - timedelta(seconds=15)),
                ("12 minutes ago", now - timedelta(minutes=12)),
                ("2 hours ago", now - timedelta(hours=2)),
                ("30 hours ago (Yesterday)", now - timedelta(hours=30)),
                ("4 days ago", now - timedelta(days=4)),
                ("18 days ago (2 weeks ago)", now - timedelta(days=18)),
            ]

            time_rows = []
            for desc, sample_time in time_samples:
                time_rows.append({
                    "Scenario": desc,
                    "Timestamp": sample_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Relative Output": format_relative_time(sample_time),
                })
            st.dataframe(pd.DataFrame(time_rows), use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 6: Tables & Empty States
    # ─────────────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("### Enterprise Data Table")
        st.caption("Subtle borders, dark background `#1e293b`, hover highlights, and tabular numbers.")

        table_html = f"""
        <table class="nivara-table">
          <thead>
            <tr>
              <th>Asset Identifier</th>
              <th>Category</th>
              <th>Status</th>
              <th>Severity</th>
              <th style="text-align: right;">Active Draw</th>
              <th style="text-align: right;">Est. Monthly INR</th>
              <th>Service Guard</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>BLD-A-CH1</code></td>
              <td>Chiller Plant</td>
              <td>{status_chip('active')}</td>
              <td>{severity_chip('low')}</td>
              <td class="numeric">142.5 kW</td>
              <td class="numeric">{format_inr(482000)}</td>
              <td>{service_safe_chip('pass')}</td>
            </tr>
            <tr>
              <td><code>BLD-B-AHU3</code></td>
              <td>HVAC Air Handler</td>
              <td>{status_chip('in_progress')}</td>
              <td>{severity_chip('high')}</td>
              <td class="numeric">38.2 kW</td>
              <td class="numeric">{format_inr(127500)}</td>
              <td>{service_safe_chip('needs_review', reason='High zone CO2 concentration.')}</td>
            </tr>
            <tr>
              <td><code>PUMP-UTL-01</code></td>
              <td>Water Treatment STP</td>
              <td>{status_chip('verified')}</td>
              <td>{severity_chip('critical')}</td>
              <td class="numeric">18.0 kW</td>
              <td class="numeric">{format_inr(56000)}</td>
              <td>{service_safe_chip('blocked', reason='Critical water recirculation loop in progress.', safe_alternative='Run secondary backup pump PUMP-UTL-02')}</td>
            </tr>
          </tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("### Standard Empty State Component")
        st.caption("Clear empty states with calm iconography and an actionable next step.")

        empty_state(
            title="No Active High-Priority Anomalies",
            description="All monitored electrical feeders, transformers, and chillers are operating within 95% confidence tolerance baselines for the current campus occupancy cycle.",
            action_label="Run Ad-hoc Telemetry Diagnostic",
            icon="✓",
            key="styleguide_demo_empty_state_btn",
        )
