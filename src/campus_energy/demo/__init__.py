"""Nivara Demo & Live Fault Injection System.

Provides canonical fault injection, active fault registry, and 6-step guided jury tour.
"""

from campus_energy.demo.inject import (
    FAULT_TYPES,
    ActiveFaultRegistry,
    active_fault_registry,
    clear_fault_registry,
    get_active_faults,
    get_fault_presets,
    get_supported_fault_types,
    inject_live_fault,
    resolve_fault,
)
from campus_energy.demo.tour import (
    TOUR_STEPS,
    GuidedTour,
    get_default_tour,
    get_tour_step_info,
)

__all__ = [
    "FAULT_TYPES",
    "TOUR_STEPS",
    "ActiveFaultRegistry",
    "GuidedTour",
    "active_fault_registry",
    "clear_fault_registry",
    "get_active_faults",
    "get_default_tour",
    "get_fault_presets",
    "get_supported_fault_types",
    "get_tour_step_info",
    "inject_live_fault",
    "resolve_fault",
]
