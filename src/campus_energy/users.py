"""User Management & Profile Authentication for Nivara Platform.

Provides Pydantic UserProfile model, configuration loading from YAML,
and authentication lookup for operational roles.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ROLE_DISPLAY_NAMES = {
    "administrator": "Administrator",
    "operations": "Operations",
    "sustainability": "Sustainability",
    "viewer": "Viewer",
}

ROLE_ICONS = {
    "administrator": "👔",
    "operations": "🔧",
    "sustainability": "🌱",
    "viewer": "👁️",
}


class UserProfile(BaseModel):
    """Institutional user profile with operational role, title, and facility scopes."""

    id: str = Field(description="Unique system user ID (e.g. usr_admin_01)")
    name: str = Field(description="Full personal name (e.g. Dr. Arun Patnaik)")
    email: str = Field(description="Institutional email address")
    role: str = Field(default="administrator", description="Assigned role: administrator, operations, sustainability")
    title: str = Field(description="Institutional job title or designation")
    facilities: list[str] = Field(
        default_factory=lambda: ["university"],
        description="List of facility IDs the user is authorized to access",
    )
    default_facility: str = Field(default="university", description="Initial landing facility ID")
    avatar_url: Optional[str] = Field(default=None, description="Optional profile image URI")

    @property
    def role_display(self) -> str:
        """User-friendly display name of the assigned role."""
        return ROLE_DISPLAY_NAMES.get(self.role.lower(), self.role.capitalize())

    @property
    def role_icon(self) -> str:
        """Icon glyph representing the role."""
        return ROLE_ICONS.get(self.role.lower(), "👤")

    @property
    def initials(self) -> str:
        """2-letter uppercase initials for avatar display."""
        parts = self.name.strip().split()
        if not parts:
            return "U"
        if len(parts) == 1:
            return parts[0][:2].upper()
        # Skip titles like Dr., Prof. if more parts exist
        if parts[0].lower().rstrip(".") in {"dr", "prof", "mr", "ms", "mrs"} and len(parts) >= 3:
            return f"{parts[1][0]}{parts[-1][0]}".upper()
        return f"{parts[0][0]}{parts[-1][0]}".upper()

    def has_facility_access(self, facility_id: str) -> bool:
        """Check whether user has authorization to access a given facility."""
        if not self.facilities:
            return True
        return facility_id.lower() in [f.lower() for f in self.facilities]

    def __getitem__(self, item: str) -> Any:
        """Backwards compatibility for dict-like subscripting."""
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)


def _resolve_users_path(path: Union[Path, str] = "config/users.yaml") -> Path:
    """Resolve user configuration path supporting relative paths and alternates."""
    p = Path(path)
    if p.is_file():
        return p.resolve()

    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / p
    if candidate.is_file():
        return candidate.resolve()

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


_USERS_CACHE: Optional[dict[str, UserProfile]] = None


def load_users(path: Union[Path, str] = "config/users.yaml") -> dict[str, UserProfile]:
    """Load user profiles from YAML into a dictionary of UserProfile objects."""
    global _USERS_CACHE
    resolved_path = _resolve_users_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Users configuration file not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    raw_users = data.get("users", data)
    users_dict: dict[str, UserProfile] = {}

    for k, v in raw_users.items():
        if isinstance(v, dict):
            item_data = dict(v)
            if "id" not in item_data or not item_data["id"]:
                item_data["id"] = f"usr_{k}"
            users_dict[k] = UserProfile(**item_data)

    _USERS_CACHE = users_dict
    return users_dict


def authenticate_user(email_or_id: str, path: Union[Path, str] = "config/users.yaml") -> Optional[UserProfile]:
    """Authenticate and retrieve a UserProfile by user ID, email, role key, or name."""
    if not email_or_id or not str(email_or_id).strip():
        return None

    query = str(email_or_id).strip().lower()
    users = load_users(path)

    # 1. Exact match on dictionary key (e.g. "admin", "operations", "sustainability")
    if query in users:
        return users[query]

    # 2. Match on user.id or user.email
    for user in users.values():
        if user.id.lower() == query:
            return user
        if user.email.lower() == query:
            return user

    # 3. Match on user.name
    for user in users.values():
        if user.name.lower() == query:
            return user

    return None


def get_demo_users(path: Union[Path, str] = "config/users.yaml") -> list[UserProfile]:
    """Return all configured demo users as a list."""
    return list(load_users(path).values())


def get_user_by_id(user_id: str, path: Union[Path, str] = "config/users.yaml") -> Optional[UserProfile]:
    """Lookup a user specifically by system user ID."""
    return authenticate_user(user_id, path=path)
