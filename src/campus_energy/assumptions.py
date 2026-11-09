"""Assumptions Registry for Nivara Campus Energy Intelligence Platform.

Centralized, validated, and traceable single source of truth for all engineering,
regulatory, financial, and environmental constants.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ConfidenceLevel = Literal["high", "medium", "low", "needs_verification"]

_SENTINEL = object()


class Assumption(BaseModel):
    """Canonical model for a defensible operational assumption."""

    key: str = Field(description="Dot-delimited identifier (e.g. 'tariff.electricity_commercial')")
    value: Any = Field(description="Typed or numeric constant value")
    unit: str = Field(description="Engineering or financial unit (e.g. 'INR/kWh', 'kgCO2e/kWh')")
    description: str = Field(description="Human-readable explanation of what this assumption represents")
    source: str = Field(description="Documented authority, standard, code, or published order")
    source_date: str = Field(description="Publication or effective date of the reference source (YYYY-MM-DD)")
    confidence: ConfidenceLevel = Field(description="Data confidence rating")
    applies_to: list[str] = Field(default_factory=lambda: ["all"], description="Applicable facility types or ['all']")
    notes: str | None = Field(default=None, description="Optional verification instructions or context")


class AssumptionsRegistry(BaseModel):
    """In-memory collection and lookup engine for validated assumptions."""

    assumptions: dict[str, Assumption] = Field(default_factory=dict)

    def get(self, key: str) -> Assumption:
        """Retrieve an Assumption record by its unique key.

        Raises:
            KeyError: If key is not present in the registry.
        """
        if key not in self.assumptions:
            raise KeyError(
                f"Assumption key '{key}' not found in registry. "
                f"Available keys count: {len(self.assumptions)}."
            )
        return self.assumptions[key]

    def get_value(self, key: str, default: Any = _SENTINEL) -> Any:
        """Retrieve the raw value of an assumption with optional fallback.

        Args:
            key: Dot-delimited identifier.
            default: Value to return if key is missing. If not provided and key is missing,
                raises KeyError.

        Returns:
            The assumption value (or default).
        """
        if key in self.assumptions:
            return self.assumptions[key].value
        if default is not _SENTINEL:
            return default
        raise KeyError(
            f"Assumption key '{key}' not found in registry and no default was provided."
        )

    def list_by_prefix(self, prefix: str) -> list[Assumption]:
        """Return all assumptions whose keys begin with the given prefix."""
        return [item for key, item in self.assumptions.items() if key.startswith(prefix)]

    def all(self) -> list[Assumption]:
        """Return all assumptions as a flat list."""
        return list(self.assumptions.values())

    def keys(self) -> list[str]:
        """Return all registered keys."""
        return list(self.assumptions.keys())

    def __getitem__(self, key: str) -> Assumption:
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.assumptions

    def __len__(self) -> int:
        return len(self.assumptions)


def _resolve_config_path(path: str | Path) -> Path:
    """Resolve assumption config file path across diverse run environments."""
    p = Path(path)
    if p.is_file():
        return p.resolve()

    # Search relative to repository root (assumptions.py -> campus_energy -> src -> root)
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / path
    if candidate.is_file():
        return candidate.resolve()

    # Support interchangeable config / configs folder names
    path_str = str(path).replace("\\", "/")
    if "config/" in path_str:
        alt = repo_root / path_str.replace("config/", "configs/")
        if alt.is_file():
            return alt.resolve()
    elif "configs/" in path_str:
        alt = repo_root / path_str.replace("configs/", "config/")
        if alt.is_file():
            return alt.resolve()

    return candidate.resolve()


@lru_cache(maxsize=16)
def load_assumptions(path: str = "config/assumptions.yaml") -> AssumptionsRegistry:
    """Load, validate, and cache assumptions from a YAML file.

    Args:
        path: Path to assumptions YAML file (relative or absolute).

    Returns:
        AssumptionsRegistry instance containing all parsed Assumption objects.
    """
    resolved_path = _resolve_config_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Assumptions config file not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_list: list[dict[str, Any]]
    if isinstance(data, dict):
        raw_list = data.get("assumptions", [])
    elif isinstance(data, list):
        raw_list = data
    else:
        raw_list = []

    registry_dict: dict[str, Assumption] = {}
    for entry in raw_list:
        model = Assumption.model_validate(entry)
        registry_dict[model.key] = model

    return AssumptionsRegistry(assumptions=registry_dict)


# Module-level shortcuts / singleton proxy functions
def get(key: str) -> Assumption:
    """Shortcut: Retrieve Assumption model from default registry."""
    return load_assumptions().get(key)


def get_value(key: str, default: Any = _SENTINEL) -> Any:
    """Shortcut: Retrieve raw assumption value from default registry."""
    return load_assumptions().get_value(key, default=default)


def list_by_prefix(prefix: str) -> list[Assumption]:
    """Shortcut: List assumptions starting with prefix from default registry."""
    return load_assumptions().list_by_prefix(prefix)


def all() -> list[Assumption]:
    """Shortcut: List all assumptions from default registry."""
    return load_assumptions().all()
