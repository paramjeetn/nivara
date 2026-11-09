"""Facility Profiles & Protected Loads for Nivara Campus Energy Intelligence Platform.

Encodes operational constraints, thermal comfort boundaries, mission-critical non-curtailable
protected loads, tariff profiles, and compliance standards across distinct facility types
(e.g., University, District Hospital, Government Office, Industrial Campus).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field


class ComfortBound(BaseModel):
    """Thermal comfort boundary constraints for a facility zone or space type."""

    min_pmv: float = Field(description="Minimum Predicted Mean Vote (PMV) lower comfort limit")
    max_pmv: float = Field(description="Maximum Predicted Mean Vote (PMV) upper comfort limit")
    temp_min_c: float = Field(description="Minimum acceptable temperature in Celsius")
    temp_max_c: float = Field(description="Maximum acceptable temperature in Celsius")


class ProtectedLoad(BaseModel):
    """Definition of a mission-critical, non-curtailable electrical or thermal asset."""

    asset_id: str = Field(description="Unique asset identifier matching telemetry/asset registries")
    asset_name: str = Field(description="Human-readable asset description")
    zone_id: Optional[str] = Field(default=None, description="Physical zone containing the asset")
    category: str = Field(
        description="Criticality class (e.g. 'life_safety', 'cold_chain', 'critical_it', 'research_lab', 'continuous_process')"
    )
    curtailable: bool = Field(
        default=False,
        description="Whether this load can be curtailed or shed during peak shaving / demand response"
    )
    min_power_kw: Optional[float] = Field(
        default=None,
        description="Absolute minimum operational electrical power in kW"
    )
    reason: str = Field(
        description="Defensible clinical, safety, regulatory, or operational reason for non-curtailability"
    )


class FacilityProfile(BaseModel):
    """Institutional facility configuration encoding operational constraints and standards."""

    facility_id: str = Field(description="Unique facility identifier slug (e.g. 'university', 'district_hospital')")
    facility_name: str = Field(description="Full official name of the facility")
    facility_type: str = Field(description="Institutional category (e.g. 'university', 'hospital', 'govt_office', 'industrial')")
    location: str = Field(description="Geographic location and state")
    floor_area_m2: float = Field(description="Total conditioned/built-up floor area in square meters")
    peak_occupancy: int = Field(description="Peak designed human occupancy")
    data_dir: str = Field(description="Path to telemetry and contextual data directory")
    comfort_bounds: dict[str, ComfortBound] = Field(
        default_factory=dict,
        description="Mapping of zone types (e.g. 'icu', 'ot', 'classrooms', 'general') to comfort bounds"
    )
    protected_loads: list[ProtectedLoad] = Field(
        default_factory=list,
        description="List of mission-critical protected loads with non-curtailable semantics"
    )
    tariff_profile: str = Field(description="Reference utility tariff schedule identifier")
    compliance_frameworks: list[str] = Field(
        default_factory=list,
        description="Applicable statutory and regulatory frameworks (e.g. BEE Star Rating, NABH, ECBC 2017)"
    )
    default_roles: list[str] = Field(
        default_factory=list,
        description="Authorized role personas (e.g. administrator, medical_superintendent, operations)"
    )
    icon: str = Field(default="🏢", description="Dashboard display emoji or icon")
    config: Optional[str] = Field(default=None, description="Path to simulation config file")

    def is_asset_protected(self, asset_id: str) -> bool:
        """Return True if asset_id is designated as a protected load."""
        return self.get_protected_load(asset_id) is not None

    def get_protected_load(self, asset_id: str) -> Optional[ProtectedLoad]:
        """Return the ProtectedLoad record for asset_id, or None if not protected."""
        if not asset_id:
            return None
        target = asset_id.strip()
        for load in self.protected_loads:
            if load.asset_id == target:
                return load
        # Case-insensitive fallback
        target_lower = target.lower()
        for load in self.protected_loads:
            if load.asset_id.lower() == target_lower:
                return load
        return None

    def get_comfort_bound_for_zone(self, zone_type: str) -> ComfortBound:
        """Retrieve thermal comfort bounds for a zone or room type with smart fallback.

        Args:
            zone_type: Zone identifier or category (e.g. 'icu', 'Classroom 101', 'labs', 'office').

        Returns:
            ComfortBound configured for that zone or a defensible general default.
        """
        if not zone_type:
            return self._fallback_comfort_bound()

        # 1. Exact match
        if zone_type in self.comfort_bounds:
            return self.comfort_bounds[zone_type]

        # 2. Case-insensitive exact match
        zt_lower = zone_type.strip().lower()
        for k, v in self.comfort_bounds.items():
            if k.lower() == zt_lower:
                return v

        # 3. Substring match (e.g. 'BLD-A-CR101' matches 'classroom', 'ICU 1' matches 'icu')
        for k, v in self.comfort_bounds.items():
            k_lower = k.lower()
            if k_lower in zt_lower or zt_lower in k_lower:
                return v

        # 4. Common semantic aliases
        alias_map = {
            "classroom": ["classrooms", "academic", "cr"],
            "lab": ["labs", "laboratory"],
            "ward": ["wards", "inpatient", "gw"],
            "waiting": ["public_waiting", "opd", "waiting_area", "wr"],
            "office": ["offices", "admin", "of"],
            "icu": ["intensive_care", "icu"],
            "ot": ["operation_theatre", "surgery", "ot"],
        }
        for std_key, aliases in alias_map.items():
            if any(a in zt_lower for a in aliases):
                for k, v in self.comfort_bounds.items():
                    if k.lower() == std_key or any(a in k.lower() for a in aliases):
                        return v

        # 5. Fallback to general, office, or first entry
        if "general" in self.comfort_bounds:
            return self.comfort_bounds["general"]
        if "office" in self.comfort_bounds:
            return self.comfort_bounds["office"]
        if self.comfort_bounds:
            return next(iter(self.comfort_bounds.values()))

        return self._fallback_comfort_bound()

    def _fallback_comfort_bound(self) -> ComfortBound:
        return ComfortBound(
            min_pmv=-0.5,
            max_pmv=0.5,
            temp_min_c=22.0,
            temp_max_c=26.0,
        )

    def __getitem__(self, item: str) -> Any:
        """Enable backwards-compatible dict-like subscripting."""
        if item == "key":
            return self.facility_type
        if hasattr(self, item):
            val = getattr(self, item)
            if item == "data_dir" and isinstance(val, str):
                return Path(val)
            return val
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        """Enable backwards-compatible dict-like get()."""
        try:
            return self[item]
        except KeyError:
            return default


class ProfileDict(dict):
    """Dictionary of facility profiles with flexible multi-key lookup."""

    def __getitem__(self, key: str) -> FacilityProfile:
        if key in self:
            return super().__getitem__(key)
        for k, prof in self.items():
            if prof.facility_id == key or prof.facility_type == key:
                return prof
        raise KeyError(f"Facility profile '{key}' not found. Available profiles: {list(self.keys())}")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _resolve_profile_path(path: Path | str) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / p
    if candidate.is_file():
        return candidate.resolve()
    path_str = str(p).replace("\\", "/")
    if "config/" in path_str:
        alt = repo_root / path_str.replace("config/", "configs/")
        if alt.is_file():
            return alt.resolve()
    elif "configs/" in path_str:
        alt = repo_root / path_str.replace("configs/", "config/")
        if alt.is_file():
            return alt.resolve()
    return candidate.resolve()


def _resolve_profile_dir(path: Path | str) -> Path:
    p = Path(path)
    if p.is_dir():
        return p.resolve()
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidate = repo_root / p
    if candidate.is_dir():
        return candidate.resolve()
    path_str = str(p).replace("\\", "/")
    if "config/" in path_str:
        alt = repo_root / path_str.replace("config/", "configs/")
        if alt.is_dir():
            return alt.resolve()
    elif "configs/" in path_str:
        alt = repo_root / path_str.replace("configs/", "config/")
        if alt.is_dir():
            return alt.resolve()
    return candidate.resolve()


def load_facility_profile(profile_path: Path | str) -> FacilityProfile:
    """Load and validate a single facility profile YAML file."""
    path = _resolve_profile_path(profile_path)
    if not path.is_file():
        raise FileNotFoundError(f"Facility profile YAML not found at: {profile_path} (resolved: {path})")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return FacilityProfile(**data)


def load_all_profiles(profiles_dir: Path | str = "config/facility_profiles") -> dict[str, FacilityProfile]:
    """Discover, load, and validate all facility profile YAML files in a directory."""
    dir_path = _resolve_profile_dir(profiles_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Facility profiles directory not found: {profiles_dir} (resolved: {dir_path})")

    profiles: dict[str, FacilityProfile] = ProfileDict()
    yaml_files = sorted(list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml")))
    for file_path in yaml_files:
        prof = load_facility_profile(file_path)
        profiles[prof.facility_id] = prof
    return profiles
