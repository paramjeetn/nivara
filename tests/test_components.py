"""Unit tests for Nivara Step UX-3: Design System, Tokens, and Component Library."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
try:
    import tomllib
except ImportError:
    import tomli as tomllib

from campus_energy.ui.components import (
    empty_state,
    format_inr,
    format_inr_compact,
    format_relative_time,
    get_kpi_card_html,
    page_heading,
    service_safe_chip,
    severity_chip,
    status_chip,
)
from campus_energy.ui.styles import get_global_css, load_global_styles
from campus_energy.ui.styleguide import render_styleguide


# ─────────────────────────────────────────────────────────────────────────────
# 1. Theme Configuration & Global Styles Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_streamlit_config_theme():
    """Verify that .streamlit/config.toml exists and defines the enterprise dark theme tokens."""
    config_path = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
    assert config_path.exists(), f"Missing config file at {config_path}"

    content = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(content)

    assert "theme" in config
    theme = config["theme"]
    assert theme["primaryColor"] == "#0284c7"
    assert theme["backgroundColor"] == "#0f172a"
    assert theme["secondaryBackgroundColor"] == "#1e293b"
    assert theme["textColor"] == "#f8fafc"
    assert theme["font"] == "sans serif"


def test_global_css_tokens():
    """Verify that global CSS contains 12-column grid, tabular-nums, and component classes."""
    css = get_global_css()
    assert len(css) > 500
    assert "--nv-bg-canvas: #0f172a;" in css
    assert "--nv-bg-surface: #1e293b;" in css
    assert "--nv-brand-primary: #0284c7;" in css
    assert "tabular-nums" in css
    assert ".nivara-grid" in css
    assert ".col-12" in css
    assert ".kpi-card" in css
    assert ".chip-status-active" in css
    assert ".chip-severity-critical" in css
    assert ".chip-service-safe-blocked" in css


# ─────────────────────────────────────────────────────────────────────────────
# 2. Indian Currency Formatting (INR) Tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("amount,precision,compact,expected", [
    (127500, 0, False, "₹1,27,500"),
    (750, 0, False, "₹750"),
    (1000, 0, False, "₹1,000"),
    (1250000, 0, False, "₹12,50,000"),
    (1250000, 1, True, "₹12.5 Lakh"),
    (14000000, 1, True, "₹1.4 Crore"),
    (14000000, 0, False, "₹1,40,00,000"),
    (-25000, 0, False, "-₹25,000"),
    (-1250000, 1, True, "-₹12.5 Lakh"),
    (0, 0, False, "₹0"),
    (54200.75, 2, False, "₹54,200.75"),
    (127500.5, 2, False, "₹1,27,500.50"),
])
def test_format_inr(amount, precision, compact, expected):
    """Verify Indian number system grouping and compact Lakh/Crore conversions."""
    result = format_inr(amount, precision=precision, compact=compact)
    assert result == expected


def test_format_inr_compact_helper():
    """Verify format_inr_compact shortcut helper."""
    assert format_inr_compact(1250000) == "₹12.5 Lakh"
    assert format_inr_compact(14000000) == "₹1.4 Crore"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Relative Human Time Formatting Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_format_relative_time():
    """Verify relative human timestamps across seconds, minutes, hours, and days."""
    ref = datetime(2026, 11, 10, 12, 0, 0)

    # 15 seconds ago -> Just now
    t_now = ref - timedelta(seconds=15)
    assert format_relative_time(t_now, reference_dt=ref) == "Just now"

    # 12 minutes ago
    t_min = ref - timedelta(minutes=12)
    assert format_relative_time(t_min, reference_dt=ref) == "12 min ago"

    # 2 hours ago
    t_hours = ref - timedelta(hours=2)
    assert format_relative_time(t_hours, reference_dt=ref) == "2 hours ago"

    # 30 hours ago -> Yesterday
    t_yesterday = ref - timedelta(hours=30)
    assert format_relative_time(t_yesterday, reference_dt=ref) == "Yesterday"

    # 4 days ago
    t_days = ref - timedelta(days=4)
    assert format_relative_time(t_days, reference_dt=ref) == "4 days ago"

    # 18 days ago -> 2 weeks ago
    t_weeks = ref - timedelta(days=18)
    assert format_relative_time(t_weeks, reference_dt=ref) == "2 weeks ago"


def test_format_relative_time_iso_string_and_timestamp():
    """Verify relative time formatting parses ISO strings and numeric unix timestamps."""
    ref = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
    t_iso = "2026-11-10T11:48:00Z"
    assert format_relative_time(t_iso, reference_dt=ref) == "12 min ago"

    # Numeric unix timestamp
    ts_val = ref.timestamp() - 120
    assert "min ago" in format_relative_time(ts_val, reference_dt=ref)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Status Chips & Severity Chips Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_status_chip():
    """Verify status chips render valid HTML with expected semantic classes and labels."""
    expected_statuses = [
        ("active", "Active", "chip-status-active"),
        ("in_progress", "In Progress", "chip-status-in_progress"),
        ("done", "Done", "chip-status-done"),
        ("verified", "Verified", "chip-status-verified"),
        ("proposed", "Proposed", "chip-status-proposed"),
        ("snoozed", "Snoozed", "chip-status-snoozed"),
        ("closed", "Closed", "chip-status-closed"),
    ]

    for status_key, expected_label, expected_class in expected_statuses:
        html_chip = status_chip(status_key)
        assert expected_class in html_chip
        assert expected_label in html_chip
        assert "chip-dot" in html_chip


def test_severity_chip():
    """Verify severity chips render appropriate icons, colors, and classes."""
    expected_severities = [
        ("critical", "Critical", "chip-severity-critical", "●"),
        ("high", "High", "chip-severity-high", "▲"),
        ("medium", "Medium", "chip-severity-medium", "■"),
        ("low", "Low", "chip-severity-low", "○"),
    ]

    for sev_key, expected_label, expected_class, expected_icon in expected_severities:
        html_chip = severity_chip(sev_key)
        assert expected_class in html_chip
        assert expected_label in html_chip
        assert expected_icon in html_chip


# ─────────────────────────────────────────────────────────────────────────────
# 5. Service-Safe Recommendation Chip Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_service_safe_chip():
    """Verify ServiceGuard recommendation chips render pass, needs_review, and blocked."""
    # 1. Pass status
    chip_pass = service_safe_chip("pass")
    assert "chip-service-safe-pass" in chip_pass
    assert "Safe to act" in chip_pass
    assert "✓" in chip_pass

    # 2. Needs review status with reason
    chip_review = service_safe_chip(
        status="needs_review",
        reason="Zone occupancy is 88%; cooling reduction may violate comfort.",
    )
    assert "chip-service-safe-needs_review" in chip_review
    assert "Needs review" in chip_review
    assert "Zone occupancy is 88%" in chip_pass or "Zone occupancy is 88%" in chip_review

    # 3. Blocked status with reason and safe alternative
    chip_blocked = service_safe_chip(
        status="blocked",
        reason="Chiller supplies critical ICU / Server loads.",
        safe_alternative="Throttle Administrative AHU-3 instead.",
    )
    assert "chip-service-safe-blocked" in chip_blocked
    assert "Not recommended" in chip_blocked
    assert "✕" in chip_blocked
    assert "Chiller supplies critical ICU" in chip_blocked
    assert "Throttle Administrative AHU-3 instead" in chip_blocked


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dense KPI Card & Page Heading Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_get_kpi_card_html():
    """Verify dense KPI card HTML generation with label, value, delta, and subtitle."""
    card_html = get_kpi_card_html(
        label="Active Load",
        value="482.4 kW",
        delta="-12.3 kW vs baseline",
        delta_color="normal",
        subtitle="Baseline: 494.7 kW",
        help_text="Real-time campus demand",
    )
    assert "kpi-card" in card_html
    assert "Active Load" in card_html
    assert "482.4 kW" in card_html
    assert "-12.3 kW vs baseline" in card_html
    assert "delta-negative" in card_html  # Starts with '-', delta_color='normal'
    assert "Baseline: 494.7 kW" in card_html
    assert "Real-time campus demand" in card_html


def test_get_kpi_card_html_inverse_delta():
    """Verify delta_color='inverse' inverts positive/negative coloration (e.g. power surge)."""
    card_html = get_kpi_card_html(
        label="Peak Exceedance",
        value="612.0 kVA",
        delta="+32.0 kVA exceedance",
        delta_color="inverse",
    )
    assert "delta-negative" in card_html  # Surcharge increase is red (negative) in inverse mode


# ─────────────────────────────────────────────────────────────────────────────
# 7. Styleguide Dev Page Execution Test
# ─────────────────────────────────────────────────────────────────────────────

def test_render_styleguide_executes_without_error():
    """Verify render_styleguide runs through all component showcases cleanly."""
    try:
        render_styleguide()
    except Exception as e:
        pytest.fail(f"render_styleguide() raised an unexpected exception: {e}")
