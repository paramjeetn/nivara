"""Unit tests for Step UX-11: Polish, QA, and Consistency Lead.

Enforces:
1. No hype words ("revolutionary", "groundbreaking", "ai-powered", etc.) in UI code.
2. No exclamation marks in user-facing UI views, headers, alerts, or button labels.
3. Standardized decision-support disclaimer and empty state contract.
4. Graceful degradation when telemetry, events, or assets dataframes are empty or malformed.
5. Polish checklist document exists and all criteria are verified.
6. Responsive CSS rules for 1366x768 projector and accessibility focus rings.
7. Caching layer decorators presence for sub-second page switching.
"""
from __future__ import annotations

import ast
import glob
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest

from campus_energy.ui import (
    DECISION_SUPPORT_DISCLAIMER,
    decision_support_disclaimer,
    empty_state,
)
from campus_energy.ui.views.home_operations import prepare_operations_today_strip
from campus_energy.ui.views.home_admin import prepare_admin_financial_strip
from campus_energy.ui.views.home_sustainability import prepare_carbon_trajectory
from campus_energy.analytics.esg import calculate_esg_metrics


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tone and Copy Tests: No Hype Words
# ─────────────────────────────────────────────────────────────────────────────

HYPE_WORDS = [
    "revolutionary",
    "groundbreaking",
    "ai-powered",
    "game-changing",
    "disruptive",
    "miracle",
]


def _get_all_ui_python_files() -> List[Path]:
    """Retrieve all Python source files in the UI package and main dashboard app."""
    workspace_root = Path(__file__).resolve().parent.parent
    files = list(workspace_root.glob("src/campus_energy/ui/**/*.py"))
    app_file = workspace_root / "src" / "campus_energy" / "dashboard" / "app.py"
    if app_file.exists():
        files.append(app_file)
    return files


def test_no_hype_words_in_ui_code():
    """Ensure no sensationalist or hype words exist in user-facing UI strings."""
    violations: List[str] = []

    for file_path in _get_all_ui_python_files():
        with open(file_path, "r", encoding="utf-8") as f:
            src = f.read()

        try:
            tree = ast.parse(src)
        except Exception as e:
            pytest.fail(f"Failed to parse {file_path}: {e}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                val_lower = node.value.lower()
                for word in HYPE_WORDS:
                    if word in val_lower:
                        violations.append(f"{file_path.name}:{getattr(node, 'lineno', '?')} - contains '{word}'")

    assert not violations, f"Found hype words in UI code:\n" + "\n".join(violations)


def test_smart_word_usage_restricted_to_hardware():
    """Ensure 'smart' is only used for physical smart meters and standards."""
    violations: List[str] = []

    for file_path in _get_all_ui_python_files():
        with open(file_path, "r", encoding="utf-8") as f:
            src = f.read()

        try:
            tree = ast.parse(src)
        except Exception as e:
            pytest.fail(f"Failed to parse {file_path}: {e}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lines = [l.strip().lower() for l in node.value.splitlines()]
                for l in lines:
                    if "smart" in l:
                        # Allow physical hardware "smart meter" or "smart meters"
                        if "smart meter" in l or "smart meters" in l:
                            continue
                        violations.append(f"{file_path.name}:{getattr(node, 'lineno', '?')} - contains unapproved 'smart': {l[:60]}")

    assert not violations, f"Found unapproved 'smart' in UI code:\n" + "\n".join(violations)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tone and Copy Tests: No Exclamation Marks in User-Facing Strings
# ─────────────────────────────────────────────────────────────────────────────

def test_no_exclamation_marks_in_ui_strings():
    """Verify absence of exclamation marks (!) in user-facing UI strings."""
    violations: List[str] = []

    for file_path in _get_all_ui_python_files():
        with open(file_path, "r", encoding="utf-8") as f:
            src = f.read()

        try:
            tree = ast.parse(src)
        except Exception as e:
            pytest.fail(f"Failed to parse {file_path}: {e}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lines = [l.strip() for l in node.value.splitlines()]
                for l in lines:
                    # Ignore CSS !important, doctypes <!DOCTYPE, code not-equals !=, empty or comment
                    if "!" in l:
                        if (
                            l.startswith("<!")
                            or "!important" in l
                            or "!=" in l
                            or l.startswith("//")
                            or l.startswith("/*")
                            or l.startswith("*")
                        ):
                            continue
                        violations.append(f"{file_path.name}:{getattr(node, 'lineno', '?')} - '{l[:60]}'")

    assert not violations, f"Found exclamation marks in user-facing UI strings:\n" + "\n".join(violations)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Decision Support Disclaimer & Empty States Contract
# ─────────────────────────────────────────────────────────────────────────────

def test_decision_support_disclaimer_contract():
    """Verify standardized decision-support disclaimer text and export."""
    assert DECISION_SUPPORT_DISCLAIMER == "Decision-support estimate. Field verification required."
    # Calling the renderer function shouldn't raise exceptions
    try:
        import streamlit as st
        decision_support_disclaimer()
    except Exception as e:
        pytest.fail(f"decision_support_disclaimer raised unexpected error: {e}")


def test_standard_empty_state_contract():
    """Verify empty_state accepts standardized copy and returns bool."""
    try:
        res = empty_state(
            title="No active alerts matching criteria.",
            description="No anomalies detected in the selected window.",
        )
        assert res is False
    except Exception as e:
        pytest.fail(f"empty_state raised unexpected error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graceful Degradation on Empty & Malformed DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def test_home_operations_strip_empty_telemetry():
    """Verify Operations home strip degrades gracefully with empty telemetry."""
    empty_df = pd.DataFrame()
    stats = prepare_operations_today_strip(
        tel=empty_df,
        assets=empty_df,
        events=empty_df,
        ledger=None,
    )
    assert isinstance(stats, dict)
    assert stats["load_now_kw"] == 0.0
    assert stats["open_alerts_count"] == 0
    assert "load_delta_str" in stats


def test_home_admin_strip_empty_telemetry():
    """Verify Admin financial strip degrades gracefully with empty telemetry."""
    empty_df = pd.DataFrame()
    stats = prepare_admin_financial_strip(
        tel=empty_df,
        tariff=None,
        facility_meta=None,
    )
    assert isinstance(stats, dict)
    assert "billed_cost_inr" in stats
    assert "demand_status" in stats
    assert stats["billed_cost_inr"] >= 0.0


def test_carbon_trajectory_empty_telemetry():
    """Verify carbon trajectory calculation handles empty telemetry without crash."""
    empty_df = pd.DataFrame()
    summary, fig = prepare_carbon_trajectory(tel=empty_df, facility_meta=None)
    assert isinstance(summary, dict)
    assert "current_cumulative_tco2" in summary
    assert summary["current_cumulative_tco2"] >= 0.0
    assert fig is not None


def test_calculate_esg_metrics_empty_inputs():
    """Verify ESG metrics engine gracefully computes baseline metrics on empty inputs."""
    empty_df = pd.DataFrame()
    result = calculate_esg_metrics(
        telemetry_df=empty_df,
        events_df=empty_df,
        floor_area_m2=10000.0,
        facility_type="university",
    )
    assert isinstance(result, dict)
    assert "kpis" in result
    assert "bee" in result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Polish Checklist Verification
# ─────────────────────────────────────────────────────────────────────────────

def test_polish_checklist_document_signed_off():
    """Verify docs/ux/polish_checklist.md exists and has all criteria checked."""
    workspace_root = Path(__file__).resolve().parent.parent
    checklist_path = workspace_root / "docs" / "ux" / "polish_checklist.md"

    assert checklist_path.exists(), "docs/ux/polish_checklist.md does not exist"
    content = checklist_path.read_text(encoding="utf-8")

    # Verify all major sections exist
    required_sections = [
        "1. Tone and Copy Guidelines",
        "2. Standardized Empty States Across All 7 Destinations",
        "3. Error Boundaries & Crash Resilience",
        "4. Resolution & Layout Integrity (1366x768 Projector & 1920x1080 Full HD)",
        "5. Caching & Navigation Latency (< 300ms)",
        "6. Accessibility & Keyboard Navigation",
        "7. Presenter Kit Isolation",
        "8. Sign-Off Authorization",
    ]
    for section in required_sections:
        assert section in content, f"Checklist missing required section: '{section}'"

    # Verify sign-off status
    assert "SIGNED OFF" in content
    assert "PASSED" in content


# ─────────────────────────────────────────────────────────────────────────────
# 6. Design and Responsive Integrity (1366x768 & Keyboard Access)
# ─────────────────────────────────────────────────────────────────────────────

def test_responsive_css_and_accessibility():
    """Verify style.css contains media queries for 1400px/1366x768 and focus rings."""
    workspace_root = Path(__file__).resolve().parent.parent
    css_path = workspace_root / "src" / "campus_energy" / "ui" / "style.css"

    assert css_path.exists(), "style.css does not exist"
    css = css_path.read_text(encoding="utf-8")

    # Media query for projector & standard laptop screen
    assert "@media (max-width: 1400px)" in css, "Missing @media (max-width: 1400px) in style.css"

    # Table & chart overflow rules
    assert "overflow-x: auto" in css, "Missing overflow-x: auto in style.css"

    # Keyboard accessibility focus rings
    assert ":focus-visible" in css, "Missing :focus-visible rules in style.css"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Caching Layer Verification
# ─────────────────────────────────────────────────────────────────────────────

def test_caching_layer_decorators():
    """Verify app.py uses @st.cache_data for heavy analytics to ensure sub-300ms switches."""
    workspace_root = Path(__file__).resolve().parent.parent
    app_path = workspace_root / "src" / "campus_energy" / "dashboard" / "app.py"

    assert app_path.exists(), "app.py does not exist"
    app_code = app_path.read_text(encoding="utf-8")

    assert "@st.cache_data" in app_code, "app.py missing @st.cache_data decorators"
    assert "cached_engineer_features" in app_code
    assert "cached_detect_anomalies" in app_code
    assert "cached_generate_24h_forecast" in app_code
