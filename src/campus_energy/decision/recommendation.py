"""Recommendation and Guard Schemas for Nivara Decision Support.

Encodes strongly typed data contracts for energy recommendations, range-based savings bundles,
and Service-Impact Guard evaluation results.
"""
from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field


class SavingsRange(BaseModel):
    """Range estimate [low, expected, high] for savings metrics."""

    low: float = Field(description="Conservative lower-bound estimate")
    expected: float = Field(description="Nominal/expected best estimate")
    high: float = Field(description="Optimistic upper-bound estimate")


class SavingsBundle(BaseModel):
    """Tripartite savings bundle covering energy (kWh), cost (INR), and emissions (kgCO2e)."""

    kwh: SavingsRange = Field(description="Estimated electrical energy savings in kWh")
    inr: SavingsRange = Field(description="Estimated cost savings in Indian Rupees (INR)")
    co2_kg: SavingsRange = Field(description="Estimated greenhouse gas abatement in kg CO2e")

    @classmethod
    def from_expected(
        cls,
        kwh_expected: float,
        inr_per_kwh: float = 7.50,
        co2_per_kwh: float = 0.82,
        spread: float = 0.20,
    ) -> SavingsBundle:
        """Helper to construct a standardized bundle from expected kWh savings.

        Args:
            kwh_expected: Expected kWh savings.
            inr_per_kwh: Commercial electricity tariff in INR/kWh.
            co2_per_kwh: Grid carbon emission factor in kgCO2e/kWh.
            spread: Fractional spread for range (+/- 20% default).
        """
        k_exp = max(0.0, float(kwh_expected))
        k_low = round(k_exp * (1.0 - spread), 1)
        k_high = round(k_exp * (1.0 + spread), 1)

        inr_exp = round(k_exp * inr_per_kwh, 1)
        inr_low = round(k_low * inr_per_kwh, 1)
        inr_high = round(k_high * inr_per_kwh, 1)

        co2_exp = round(k_exp * co2_per_kwh, 1)
        co2_low = round(k_low * co2_per_kwh, 1)
        co2_high = round(k_high * co2_per_kwh, 1)

        return cls(
            kwh=SavingsRange(low=k_low, expected=k_exp, high=k_high),
            inr=SavingsRange(low=inr_low, expected=inr_exp, high=inr_high),
            co2_kg=SavingsRange(low=co2_low, expected=co2_exp, high=co2_high),
        )


class GuardResult(BaseModel):
    """Outcome of Service-Impact Guard evaluation for an energy recommendation."""

    status: Literal["pass", "blocked", "needs_review"] = Field(
        description="Enforcement status of recommendation against service constraints"
    )
    reason: str = Field(
        description="Defensible technical, clinical, or operational rationale for guard decision"
    )
    constraints_checked: List[str] = Field(
        default_factory=list,
        description="List of constraint identifiers evaluated (e.g. 'protected_load_check', 'pmv_comfort_check')",
    )
    safe_alternative: Optional[str] = Field(
        default=None,
        description="Actionable, non-harmful AI counter-proposal when action is blocked",
    )


class Recommendation(BaseModel):
    """Canonical model for an actionable energy decision-support recommendation."""

    id: str = Field(description="Unique recommendation identifier (e.g. 'REC-2026-001')")
    facility_id: str = Field(description="Target facility identifier (e.g. 'university', 'district_hospital')")
    type: str = Field(
        description="Recommendation category: 'anomaly_fix', 'setpoint_adjustment', 'schedule_shift', "
                    "'peak_shaving', 'led_upgrade', 'solar_bess', 'equipment_shutdown'"
    )
    target: str = Field(description="Target asset identifier or zone identifier")
    action_text: str = Field(description="Plain-language actionable description of the proposed energy measure")
    trigger: str = Field(description="Triggering event, anomaly alert ID, forecast condition, or scenario")
    savings: SavingsBundle = Field(description="Tripartite estimated savings bundle (kWh, INR, kg CO2e)")
    cost: float = Field(default=0.0, description="Estimated implementation capex/opex cost in INR")
    payback_months: Optional[float] = Field(
        default=None, description="Simple payback period in months if cost > 0"
    )
    assumption_refs: List[str] = Field(
        default_factory=list,
        description="Assumption keys from registry referenced during savings calculation",
    )
    service_guard: Optional[GuardResult] = Field(
        default=None,
        description="Attached Service-Impact Guard verification result",
    )
    owner_role: Literal["administrator", "operations", "sustainability"] = Field(
        default="operations",
        description="Designated institutional persona responsible for approval / execution",
    )
    lifecycle: Literal["proposed", "accepted", "work_order", "done", "verified"] = Field(
        default="proposed",
        description="Lifecycle stage of the recommendation in facility workflow",
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence score of the recommendation",
    )
    is_estimate: bool = Field(
        default=True,
        description="Mandatory decision-support disclaimer flag (estimate, not legal/official measurement)",
    )

    @property
    def is_blocked(self) -> bool:
        """Return True if service guard explicitly blocked this recommendation."""
        return self.service_guard is not None and self.service_guard.status == "blocked"

    @property
    def is_passed(self) -> bool:
        """Return True if service guard passed this recommendation."""
        return self.service_guard is not None and self.service_guard.status == "pass"
