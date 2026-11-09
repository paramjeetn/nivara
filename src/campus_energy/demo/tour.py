"""Guided Demo Tour Engine for Nivara Hackathon Jury Presentations.

Defines a structured, 6-step walkthrough tailored for a crisp 2-minute live demo:
  Step 1: 📊 Executive Baseline (Review facility baseline, tariffs, and live IoT status)
  Step 2: ⚡ Live Fault Injection (Trigger chosen fault on target asset)
  Step 3: 🚨 Anomaly Detection & "Why?" Card (Instant alert, factor attribution, multi-layer consensus)
  Step 4: 🛡️ Guarded Action (Unified recommendation with 3-point savings and Service-Impact Guard certification)
  Step 5: 🛠️ Work Order & Notification Dispatch (Dispatches CMMS ticket, sends live Telegram/Email alert)
  Step 6: 🔬 Resolution & Verified M&V Ledger (Simulates repair fix, verifies counterfactual savings, updates hit rate)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from campus_energy.demo.inject import clear_fault_registry

logger = logging.getLogger(__name__)

TOUR_STEPS: list[dict[str, Any]] = [
    {
        "step": 1,
        "title": "📊 Executive Baseline",
        "short_name": "Executive Baseline",
        "target_page": "📊  Executive Overview",
        "badge": "01: BASELINE",
        "time_budget": "20s",
        "narrative": (
            "Start here: Nivara tracks real-time campus power, carbon footprint, and tariff compliance "
            "under OERC HT Commercial rules with zero manual entry. Notice the facility baseline, "
            "active solar microgrid contribution, and verified IPMVP ledger."
        ),
        "talking_points": [
            "Unified single-pane across 25+ assets with live IoT telemetry.",
            "ODISHA OERC HT Commercial tariff integration with peak/off-peak TOD pricing.",
            "Live streaming telemetry validated with automated data-quality checks.",
        ],
        "action_buttons": ["Begin Live Demo", "Inspect Baseline"],
        "action_hint": "Click 'Next Step' to simulate an unexpected operational anomaly.",
    },
    {
        "step": 2,
        "title": "⚡ Live Fault Injection",
        "short_name": "Fault Injection",
        "target_page": "🚨  Alerts & NLP",
        "badge": "02: INJECTION",
        "time_budget": "20s",
        "narrative": (
            "Simulate an unmanaged operational crisis in real time: inject an operational or physical fault "
            "(e.g., lights left on overnight, pump cavitation dry-run, distribution pipe rupture, "
            "IoT sensor dropout, or low power factor penalty) into live telemetry."
        ),
        "talking_points": [
            "Supports 5 canonical physical, electrical, and hydraulic faults.",
            "Mutates real-time telemetry stream on target asset in milliseconds.",
            "Preserves pre-fault ground truth for automated closed-loop counterfactual M&V.",
        ],
        "action_buttons": ["🚨 Inject Fault Now", "Choose Preset"],
        "action_hint": "Select a fault type and asset, then trigger 'Inject Fault Now'.",
    },
    {
        "step": 3,
        "title": "🚨 Anomaly Detection & 'Why?' Card",
        "short_name": "Anomaly Consensus",
        "target_page": "🚨  Alerts & NLP",
        "badge": "03: DETECTION",
        "time_budget": "20s",
        "narrative": (
            "Observe instant detection: Nivara's triple-engine consensus (Physical Rules + Ridge Regression "
            "Baseline + Isolation Forest ML) detects the anomaly in milliseconds, attributes root cause factors, "
            "and generates a human-readable 'Why?' card."
        ),
        "talking_points": [
            "Multi-layer consensus eliminates false alarms while catching true failures.",
            "Transparent Factor Attribution Card explains 'Why did this alert fire?'.",
            "Calculates excess kWh waste run-rate and daily financial exposure in ₹.",
        ],
        "action_buttons": ["Inspect 'Why?' Card", "View Evidence"],
        "action_hint": "Review the multi-layer agreement score and root cause evidence.",
    },
    {
        "step": 4,
        "title": "🛡️ Guarded Action",
        "short_name": "Guarded Action",
        "target_page": "🚨  Alerts & NLP",
        "badge": "04: GUARDED ACTION",
        "time_budget": "20s",
        "narrative": (
            "AI decision support with hard safety constraints: Nivara issues a 3-point savings estimate "
            "(kWh, ₹, CO2) certified by the Service-Impact Guard. Protected research labs, exam halls, "
            "and critical hospital loads are never curtailed."
        ),
        "talking_points": [
            "Tri-vector savings bundle: energy (kWh), utility bill (₹), and Scope 2 carbon abatement.",
            "Service-Impact Guard certifies non-curtailable protected loads.",
            "Provides automated safe operational alternatives when load curtailment is blocked.",
        ],
        "action_buttons": ["Check Guard Certification", "View Recommendations"],
        "action_hint": "Notice how the Service-Impact Guard prevents disruption to protected facilities.",
    },
    {
        "step": 5,
        "title": "🛠️ Work Order & Notification Dispatch",
        "short_name": "Work Order Dispatch",
        "target_page": "🚨  Alerts & NLP",
        "badge": "05: DISPATCH",
        "time_budget": "20s",
        "narrative": (
            "Closing the loop: Turn the AI recommendation into a prioritized CMMS work order with a single click. "
            "Dynamic SLA countdowns and live alerts are dispatched simultaneously via Telegram, Email, and SMS."
        ),
        "talking_points": [
            "Automated CMMS work order assignment with SLA countdowns based on urgency.",
            "Live multi-channel notification dispatch directly to maintenance technicians.",
            "Dry-run simulation and live Telegram bot gateway support.",
        ],
        "action_buttons": ["Dispatch Work Order", "Test Notification"],
        "action_hint": "Click 'Dispatch Work Order' to send field alerts and trigger the SLA timer.",
    },
    {
        "step": 6,
        "title": "🔬 Resolution & Verified M&V Ledger",
        "short_name": "Verified M&V",
        "target_page": "🚨  Alerts & NLP",
        "badge": "06: VERIFIED M&V",
        "time_budget": "20s",
        "narrative": (
            "Auditable financial proof: Once maintenance completes the physical fix, Nivara simulates "
            "restoration, verifies counterfactual savings via IPMVP Option B, and updates the persistent "
            "Savings Ledger and recommendation hit rate in real time."
        ),
        "talking_points": [
            "IPMVP Option B counterfactual baseline compares post-fix telemetry against expected load.",
            "Persistent savings ledger maintains audited financial ROI and CO2 abatement records.",
            "Exports executive PDF & Markdown governance compliance reports.",
        ],
        "action_buttons": ["Verify Counterfactual Savings", "Complete Demo"],
        "action_hint": "Click 'Resolve & Verify' to complete the closed-loop M&V verification.",
    },
]


class GuidedTour:
    """State machine governing the 6-step guided demo walkthrough."""

    def __init__(self, step: int = 1) -> None:
        self._current_step: int = max(1, min(step, 6))
        self._active: bool = False
        self._injected_event_id: str | None = None
        self._reaction_latency_ms: float | None = None

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        self._active = True

    def deactivate(self) -> None:
        self._active = False

    def next_step(self) -> int:
        """Advance to next step, capped at 6."""
        if self._current_step < 6:
            self._current_step += 1
        return self._current_step

    def prev_step(self) -> int:
        """Return to previous step, floored at 1."""
        if self._current_step > 1:
            self._current_step -= 1
        return self._current_step

    def set_step(self, step: int) -> int:
        """Set step explicitly within [1, 6]."""
        if not (1 <= step <= 6):
            raise ValueError(f"Step must be between 1 and 6, got {step}")
        self._current_step = step
        return self._current_step

    def reset_demo(self) -> None:
        """Restore clean state: reset step to 1 and clear active faults."""
        self._current_step = 1
        self._injected_event_id = None
        self._reaction_latency_ms = None
        clear_fault_registry()

    def get_step_info(self, step: int | None = None) -> dict[str, Any]:
        """Return full metadata dictionary for specified or current step."""
        target = self._current_step if step is None else step
        if not (1 <= target <= 6):
            target = 1
        return dict(TOUR_STEPS[target - 1])

    def is_first_step(self) -> bool:
        return self._current_step == 1

    def is_last_step(self) -> bool:
        return self._current_step == 6

    def set_injected_event_id(self, event_id: str) -> None:
        self._injected_event_id = event_id

    def get_injected_event_id(self) -> str | None:
        return self._injected_event_id

    def set_reaction_latency_ms(self, latency_ms: float) -> None:
        self._reaction_latency_ms = round(latency_ms, 2)

    def get_reaction_latency_ms(self) -> float | None:
        return self._reaction_latency_ms


_DEFAULT_TOUR = GuidedTour()


def get_default_tour() -> GuidedTour:
    """Return singleton GuidedTour instance."""
    return _DEFAULT_TOUR


def get_tour_step_info(step: int) -> dict[str, Any]:
    """Retrieve metadata for a specific step number (1-6)."""
    if 1 <= step <= 6:
        return dict(TOUR_STEPS[step - 1])
    raise ValueError(f"Invalid tour step {step}; must be between 1 and 6.")
