"""Nivara Global Styling Loader and Design System Tokens.

Provides `load_global_styles()` to inject enterprise slate dark CSS,
12-column grid utilities, dense metric cards, chip badges, and tabular numbers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import streamlit as st

logger = logging.getLogger(__name__)

_STYLE_FILE = Path(__file__).parent / "style.css"


def get_global_css() -> str:
    """Return raw CSS content of Nivara enterprise stylesheet."""
    try:
        if _STYLE_FILE.exists():
            return _STYLE_FILE.read_text(encoding="utf-8")
        logger.warning("style.css not found at %s", _STYLE_FILE)
        return ""
    except Exception as e:
        logger.error("Failed to read global stylesheet: %s", e)
        return ""


def load_global_styles() -> None:
    """Inject Nivara enterprise stylesheet into the active Streamlit app session."""
    css = get_global_css()
    if css:
        st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
