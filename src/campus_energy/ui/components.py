"""Nivara Reusable Component Library.

Enterprise UI components for facility operations:
- Dense KPI Metric Cards with calculation popovers and delta badges
- Semantic Status Chips (`active`, `in_progress`, `done`, `verified`, `proposed`, `snoozed`, `closed`)
- Severity Chips (`critical`, `high`, `medium`, `low`)
- Service-Safe Recommendation Chips (`pass`, `needs_review`, `blocked` with reason/alternative tooltips)
- Indian Currency Formatter (`format_inr`, e.g. ₹1,27,500, ₹12.5 Lakh, ₹1.4 Crore)
- Relative Human Time Formatter (`format_relative_time`)
- Standardized Empty State Cards
- Calculation Traceability Info Popovers with Assumptions Registry lookup
"""
from __future__ import annotations

import html
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Indian Currency Formatting (INR)
# ─────────────────────────────────────────────────────────────────────────────

def format_inr(
    amount: Union[int, float, str],
    precision: int = 0,
    compact: bool = False,
) -> str:
    """Format numeric values according to Indian number grouping system (Lakh/Crore).

    Examples:
        format_inr(127500) -> "₹1,27,500"
        format_inr(1250000, compact=True) -> "₹12.5 Lakh"
        format_inr(14000000, compact=True) -> "₹1.4 Crore"
        format_inr(750) -> "₹750"
        format_inr(-25000) -> "-₹25,000"
        format_inr(0) -> "₹0"
    """
    try:
        val = float(amount)
    except (ValueError, TypeError):
        return f"₹{amount}"

    if abs(val) < 1e-9:
        return "₹0" if precision == 0 else f"₹0.{'0'*precision}"

    sign = "-" if val < 0 else ""
    abs_val = abs(val)

    if compact:
        if abs_val >= 10_000_000:
            cr = abs_val / 10_000_000
            p = precision if precision > 0 else (0 if cr.is_integer() else 1)
            formatted = f"{cr:.{p}f}".rstrip("0").rstrip(".") if precision == 0 else f"{cr:.{p}f}"
            return f"{sign}₹{formatted} Crore"
        elif abs_val >= 100_000:
            lakh = abs_val / 100_000
            p = precision if precision > 0 else (0 if lakh.is_integer() else 1)
            formatted = f"{lakh:.{p}f}".rstrip("0").rstrip(".") if precision == 0 else f"{lakh:.{p}f}"
            return f"{sign}₹{formatted} Lakh"

    # Standard Indian Numbering
    if precision > 0:
        formatted_num = f"{abs_val:.{precision}f}"
        int_part, dec_part = formatted_num.split(".")
    else:
        int_part = str(int(round(abs_val)))
        dec_part = ""

    if len(int_part) <= 3:
        int_formatted = int_part
    else:
        tail = int_part[-3:]
        head = int_part[:-3]
        groups = []
        while head:
            groups.append(head[-2:])
            head = head[:-2]
        groups.reverse()
        int_formatted = ",".join(groups) + "," + tail

    res = f"{sign}₹{int_formatted}"
    if dec_part:
        res += f".{dec_part}"
    return res


def format_inr_compact(amount: Union[int, float, str], precision: int = 1) -> str:
    """Convenience shortcut for compact Indian currency representation."""
    return format_inr(amount, precision=precision, compact=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Relative Human Time Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_relative_time(
    dt: Union[datetime, str, int, float],
    reference_dt: Optional[datetime] = None,
) -> str:
    """Format a timestamp into concise, human-relative operational time.

    Examples:
        - "Just now" (< 45s)
        - "12 min ago" (< 60m)
        - "2 hours ago" (< 24h)
        - "Yesterday" (24h - 48h)
        - "3 days ago" (< 7d)
        - "2 weeks ago" (>= 7d)
    """
    if isinstance(dt, (int, float)):
        target = datetime.fromtimestamp(dt, tz=timezone.utc)
    elif isinstance(dt, str):
        try:
            target = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            try:
                target = pd.to_datetime(dt).to_pydatetime()
            except Exception:
                return str(dt)
    elif isinstance(dt, datetime):
        target = dt
    else:
        return str(dt)

    ref = reference_dt or (datetime.now(timezone.utc) if target.tzinfo else datetime.now())

    # Align tz-awareness to prevent subtraction errors
    if target.tzinfo is not None and ref.tzinfo is None:
        ref = ref.replace(tzinfo=target.tzinfo)
    elif target.tzinfo is None and ref.tzinfo is not None:
        target = target.replace(tzinfo=ref.tzinfo)

    diff = ref - target
    total_seconds = diff.total_seconds()

    if total_seconds < 0:
        future_sec = abs(total_seconds)
        if future_sec < 60:
            return "in a few seconds"
        if future_sec < 3600:
            return f"in {int(future_sec // 60)} min"
        if future_sec < 86400:
            return f"in {int(future_sec // 3600)} hours"
        return "in the future"

    sec = int(total_seconds)
    if sec < 45:
        return "Just now"
    if sec < 90:
        return "1 min ago"
    if sec < 3600:
        return f"{sec // 60} min ago"
    if sec < 7200:
        return "1 hour ago"
    if sec < 86400:
        return f"{sec // 3600} hours ago"
    if sec < 172800:
        return "Yesterday"
    if sec < 86400 * 7:
        return f"{sec // 86400} days ago"
    if sec < 86400 * 30:
        weeks = max(1, sec // (86400 * 7))
        return f"{weeks} {'week' if weeks == 1 else 'weeks'} ago"

    return target.strftime("%d %b %Y")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Status Chip
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_MAP = {
    "active": ("Active", "chip-status-active"),
    "in_progress": ("In Progress", "chip-status-in_progress"),
    "done": ("Done", "chip-status-done"),
    "verified": ("Verified", "chip-status-verified"),
    "proposed": ("Proposed", "chip-status-proposed"),
    "snoozed": ("Snoozed", "chip-status-snoozed"),
    "closed": ("Closed", "chip-status-closed"),
}


def status_chip(status: str) -> str:
    """Generate an HTML badge representing operational lifecycle status.

    Supported statuses:
        `active`, `in_progress`, `done`, `verified`, `proposed`, `snoozed`, `closed`.
    """
    key = str(status).strip().lower().replace(" ", "_")
    label, css_cls = _STATUS_MAP.get(key, (str(status).title(), f"chip-status-{key}"))
    return f'<span class="chip chip-status {css_cls}"><span class="chip-dot"></span>{label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# 4. Severity Chip
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY_MAP = {
    "critical": ("Critical", "●", "chip-severity-critical"),
    "high": ("High", "▲", "chip-severity-high"),
    "medium": ("Medium", "■", "chip-severity-medium"),
    "low": ("Low", "○", "chip-severity-low"),
}


def severity_chip(severity: str) -> str:
    """Generate an HTML badge representing anomaly/alert severity.

    Supported severities:
        `critical` (red), `high` (orange), `medium` (amber), `low` (blue).
    """
    key = str(severity).strip().lower()
    label, icon, css_cls = _SEVERITY_MAP.get(key, (str(severity).title(), "•", f"chip-severity-{key}"))
    return f'<span class="chip chip-severity {css_cls}">{icon} {label}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# 5. Service-Safe Recommendation Chip
# ─────────────────────────────────────────────────────────────────────────────

_SERVICE_SAFE_MAP = {
    "pass": ("Safe to act", "✓", "chip-service-safe-pass"),
    "safe": ("Safe to act", "✓", "chip-service-safe-pass"),
    "needs_review": ("Needs review", "⚠", "chip-service-safe-needs_review"),
    "review": ("Needs review", "⚠", "chip-service-safe-needs_review"),
    "blocked": ("Not recommended", "✕", "chip-service-safe-blocked"),
    "unsafe": ("Not recommended", "✕", "chip-service-safe-blocked"),
}


def service_safe_chip(
    status: str,
    reason: Optional[str] = None,
    safe_alternative: Optional[str] = None,
) -> str:
    """Generate an HTML badge representing ServiceGuard operational safety checks.

    Statuses:
        `pass` ("Safe to act" - emerald)
        `needs_review` ("Needs review" - amber)
        `blocked` ("Not recommended" - rose/red)

    Reason and alternative are encoded in hover tooltip (`title`) and data attributes.
    """
    norm_status = str(status).strip().lower().replace(" ", "_")
    label, icon, css_cls = _SERVICE_SAFE_MAP.get(norm_status, ("Unknown", "•", "chip-service-safe-needs_review"))

    tooltip_parts = []
    if reason:
        tooltip_parts.append(f"Reason: {reason.strip()}")
    if safe_alternative:
        tooltip_parts.append(f"Safe Alternative: {safe_alternative.strip()}")

    tooltip_str = " | ".join(tooltip_parts) if tooltip_parts else f"Service Safety Check: {label}"
    info_icon = ' <span style="opacity: 0.8; font-size: 0.68rem; margin-left: 2px;">ⓘ</span>' if tooltip_parts else ""

    return (
        f'<span class="chip chip-service-safe {css_cls}" '
        f'title="{html.escape(tooltip_str)}" '
        f'data-service-safe="{norm_status}">'
        f'{icon} {label}{info_icon}'
        f'</span>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dense KPI Metric Card
# ─────────────────────────────────────────────────────────────────────────────

def get_kpi_card_html(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    subtitle: Optional[str] = None,
    help_text: Optional[str] = None,
) -> str:
    """Generate HTML string for a dense, scannable enterprise KPI card."""
    delta_html = ""
    if delta is not None:
        delta_str = str(delta).strip()
        is_neg = delta_str.startswith("-")
        is_pos = delta_str.startswith("+")

        if delta_color == "inverse":
            css_delta = "delta-negative" if is_pos else ("delta-positive" if is_neg else "delta-neutral")
        elif delta_color == "normal":
            css_delta = "delta-positive" if is_pos else ("delta-negative" if is_neg else "delta-neutral")
        else:
            css_delta = "delta-neutral"

        delta_html = f'<span class="kpi-delta {css_delta}">{html.escape(delta_str)}</span>'

    subtitle_html = f'<span class="kpi-subtitle">{html.escape(subtitle)}</span>' if subtitle else ""
    help_html = f'<span title="{html.escape(help_text)}" style="cursor: help; color: #64748b; font-size: 0.8rem;">ⓘ</span>' if help_text else ""

    return f"""
    <div class="kpi-card">
      <div class="kpi-card-header">
        <span class="kpi-label">{html.escape(label)}</span>
        {help_html}
      </div>
      <div class="kpi-value-container">
        <span class="kpi-value">{html.escape(str(value))}</span>
      </div>
      <div class="kpi-footer">
        {delta_html}
        {subtitle_html}
      </div>
    </div>
    """


def kpi_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    delta_color: str = "normal",
    subtitle: Optional[str] = None,
    help_text: Optional[str] = None,
    popover_title: Optional[str] = None,
    popover_calc_notes: Optional[str] = None,
    popover_assumptions: Optional[list[str]] = None,
) -> None:
    """Render a dense, scannable KPI metric card into the active Streamlit column/view.

    If calculation notes or assumptions are provided, renders a traceable info popover.
    """
    card_html = get_kpi_card_html(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        subtitle=subtitle,
        help_text=help_text,
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if popover_calc_notes or popover_assumptions:
        info_popover(
            title=popover_title or f"Calculation Notes — {label}",
            calculation_notes=popover_calc_notes or "Operational calculation details and traceable assumptions.",
            assumption_keys=popover_assumptions,
            button_label="ⓘ How is this calculated?",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Traceability Info Popover
# ─────────────────────────────────────────────────────────────────────────────

def info_popover(
    title: str,
    calculation_notes: str,
    assumption_keys: Optional[list[str]] = None,
    button_label: str = "ⓘ",
    help: str = "Traceable assumptions and engineering notes",
) -> None:
    """Render a non-intrusive Streamlit popover containing calculation notes & assumptions."""
    with st.popover(button_label, help=help):
        st.markdown(f"#### {title}")
        st.markdown(calculation_notes)

        if assumption_keys:
            st.divider()
            st.caption("📋 **Traceable Regulatory & Engineering Assumptions**")

            rows = []
            try:
                from campus_energy import assumptions as asm_registry

                for key in assumption_keys:
                    try:
                        item = asm_registry.get(key)
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
                            "Authority / Source": f"{item.source} ({item.source_date})",
                            "Description": item.description,
                        })
                    except KeyError:
                        rows.append({
                            "Assumption Key": key,
                            "Value": "—",
                            "Confidence": "⚪ Unregistered",
                            "Authority / Source": "Custom Variable",
                            "Description": "Dynamic runtime parameter",
                        })
            except Exception as e:
                logger.debug("Failed loading assumptions in popover: %s", e)

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Standardized Empty State Card
# ─────────────────────────────────────────────────────────────────────────────

def page_heading(
    title: str,
    subtitle: Optional[str] = None,
    badge: Optional[str] = None,
) -> None:
    """Render a clean enterprise page heading without raw decorative emojis."""
    clean_title = html.escape(title)
    sub_html = f'<p class="nivara-caption" style="margin-top: 2px;">{html.escape(subtitle)}</p>' if subtitle else ""
    badge_html = f'<span class="chip chip-status chip-status-verified">{html.escape(badge)}</span>' if badge else ""

    html_str = f"""
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 14px;">
      <div>
        <h1 class="nivara-heading-1"><span class="heading-accent"></span>{clean_title}</h1>
        {sub_html}
      </div>
      {badge_html}
    </div>
    """
    st.markdown(html_str, unsafe_allow_html=True)


# Standardized Empty State Titles
EMPTY_ACTIVE_ALERTS: str = "No active alerts matching criteria."
EMPTY_WORK_ORDERS: str = "No open work orders."
EMPTY_ANOMALIES: str = "No anomalies detected in the selected window."
EMPTY_AUDIT_LOGS: str = "No historical audit logs found."


def empty_state(
    title: str,
    description: str = "",
    action_label: Optional[str] = None,
    on_click: Optional[Callable[[], Any]] = None,
    icon: str = "🔍",
    key: Optional[str] = None,
    message: Optional[str] = None,
) -> bool:
    """Render a clean slate empty state with clear context and optional next action."""
    desc = description or message or ""
    html_content = f"""
    <div class="nivara-empty-state">
      <div class="empty-state-icon">{icon}</div>
      <div class="empty-state-title">{html.escape(title)}</div>
      <div class="empty-state-desc">{html.escape(desc)}</div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

    if action_label:
        button_key = key or f"empty_state_action_{title.lower().replace(' ', '_').replace('.', '')}"
        clicked = st.button(action_label, key=button_key, use_container_width=False)
        if clicked and on_click:
            on_click()
        return clicked

    return False


# ─────────────────────────────────────────────────────────────────────────────
# 9. Decision Support Disclaimer
# ─────────────────────────────────────────────────────────────────────────────

DECISION_SUPPORT_DISCLAIMER: str = "Decision-support estimate. Field verification required."


def decision_support_disclaimer(
    extra_text: Optional[str] = None,
    badge: bool = True,
) -> None:
    """Render the standardized decision-support disclaimer across planning & estimate views."""
    text = DECISION_SUPPORT_DISCLAIMER
    if extra_text:
        text = f"{text} {extra_text}"
    st.caption(f"ℹ️ *{html.escape(text)}*")

