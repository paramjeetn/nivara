"""Nivara Operational UI components, design tokens, and layout shell."""
from campus_energy.ui.styles import load_global_styles, get_global_css
from campus_energy.ui.components import (
    kpi_card,
    get_kpi_card_html,
    status_chip,
    severity_chip,
    service_safe_chip,
    format_inr,
    format_inr_compact,
    format_relative_time,
    empty_state,
    info_popover,
    page_heading,
)
from campus_energy.ui.top_bar import render_top_bar
from campus_energy.ui.login import render_login_screen
from campus_energy.ui.views.alerts_work import render_alerts_work

__all__ = [
    "load_global_styles",
    "get_global_css",
    "kpi_card",
    "get_kpi_card_html",
    "status_chip",
    "severity_chip",
    "service_safe_chip",
    "format_inr",
    "format_inr_compact",
    "format_relative_time",
    "empty_state",
    "info_popover",
    "page_heading",
    "render_top_bar",
    "render_login_screen",
    "render_alerts_work",
]
