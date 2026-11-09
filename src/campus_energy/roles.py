"""Role-Based Views & Personas for Nivara Campus Energy Intelligence Platform.

Defines operational roles (Administrator, Operations, Sustainability), focus widget mappings,
default view landing pages, and tailored AI assistant system prompts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import yaml
from pydantic import BaseModel, Field

VALID_FOCUS_WIDGETS: Set[str] = {
    # Administrator
    "executive_kpis",
    "tariff_summary",
    "ranked_actions",
    "ledger_totals",
    "compliance_scorecard",
    "department_billing",
    # Operations
    "live_telemetry_feed",
    "active_anomalies",
    "why_diagnostic_cards",
    "work_order_queue",
    "sla_timers",
    "service_guard_blocks",
    "data_quality",
    # Sustainability
    "esg_scorecard",
    "bee_star_rating",
    "napcc_carbon_burndown",
    "verified_impact_ledger",
    "microgrid_solar_bess",
    "verification_evidence",
}


class RoleDefinition(BaseModel):
    """Configuration and metadata for an operational dashboard role."""

    role_key: str = Field(default="", description="Unique identifier slug for role")
    name: str = Field(description="Display title for persona")
    icon: str = Field(description="Emoji or icon glyph")
    description: str = Field(description="Operational scope and responsibility summary")
    default_page: str = Field(description="Default Streamlit landing page title")
    focus_widgets: list[str] = Field(default_factory=list, description="Prioritized UI widgets")
    gemini_persona_prompt: str = Field(description="Domain-specific directive injected into Gemini system prompt")

    def has_focus_widget(self, widget_id: str) -> bool:
        """Check whether a specific widget is in this role's focus areas."""
        return widget_id in self.focus_widgets

    def __getitem__(self, item: str) -> Any:
        """Backwards compatibility for dict-like subscripting."""
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


class RolesRegistry(BaseModel):
    """Collection of role definitions supporting key-based lookup."""

    roles: dict[str, RoleDefinition] = Field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.roles.get(key, default)

    def __getitem__(self, key: str) -> RoleDefinition:
        if key not in self.roles:
            raise KeyError(f"Unknown role '{key}'. Available roles: {list(self.roles.keys())}")
        return self.roles[key]

    def __contains__(self, key: str) -> bool:
        return key in self.roles

    def __iter__(self):
        return iter(self.roles)

    def items(self):
        return self.roles.items()

    def values(self):
        return self.roles.values()

    def keys(self):
        return self.roles.keys()


def _resolve_roles_path(path: Path | str = "config/roles.yaml") -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / p
    if candidate.is_file():
        return candidate.resolve()
    # Try alternate directory (configs instead of config, or vice versa)
    p_str = str(p)
    if "config/" in p_str or "config\\" in p_str:
        alt = repo_root / Path(p_str.replace("config", "configs", 1))
        if alt.is_file():
            return alt.resolve()
    elif "configs/" in p_str or "configs\\" in p_str:
        alt = repo_root / Path(p_str.replace("configs", "config", 1))
        if alt.is_file():
            return alt.resolve()
    return p.resolve()


_ROLES_CACHE: Optional[dict[str, RoleDefinition]] = None


def load_roles(path: Path | str = "config/roles.yaml") -> dict[str, RoleDefinition]:
    """Load roles configuration from YAML file into a dictionary of RoleDefinition objects."""
    global _ROLES_CACHE
    resolved_path = _resolve_roles_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Roles configuration file not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_roles = data.get("roles", data)
    roles_dict: dict[str, RoleDefinition] = {}

    for k, v in raw_roles.items():
        if isinstance(v, dict):
            item_data = dict(v)
            if "role_key" not in item_data or not item_data["role_key"]:
                item_data["role_key"] = k
            roles_dict[k] = RoleDefinition(**item_data)

    _ROLES_CACHE = roles_dict
    return roles_dict


def get_role(role_key: str, path: Path | str = "config/roles.yaml") -> RoleDefinition:
    """Retrieve a RoleDefinition by its unique role key."""
    global _ROLES_CACHE
    if _ROLES_CACHE is None:
        _ROLES_CACHE = load_roles(path)

    if role_key not in _ROLES_CACHE:
        _ROLES_CACHE = load_roles(path)
        if role_key not in _ROLES_CACHE:
            raise KeyError(f"Unknown role '{role_key}'. Available roles: {list(_ROLES_CACHE.keys())}")

    return _ROLES_CACHE[role_key]
