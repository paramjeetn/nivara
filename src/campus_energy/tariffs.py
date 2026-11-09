"""Indian Tariff Configuration and Billing Models for Nivara Platform.

Encodes state regulatory commission tariff structures (e.g. OERC / TPCODL HT Commercial),
Time-of-Day (ToD) tariff bands, maximum demand (kVA) billing thresholds, and
power factor penalty/incentive surcharge formulations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_TARIFF_ID = "odisha_ht_commercial"


class ToDBand(BaseModel):
    """Time-of-Day (ToD) tariff band specification."""

    start_hour: Optional[int] = Field(default=None, description="Start hour of ToD window (0-23 inclusive)")
    end_hour: Optional[int] = Field(default=None, description="End hour of ToD window (0-23 exclusive)")
    multiplier: float = Field(default=1.0, description="Rate multiplier relative to normal base energy charge")
    rate_inr: float = Field(default=8.50, description="Applicable energy rate in INR per kWh for this band")


class IndianTariff(BaseModel):
    """Institutional Indian electrical utility tariff configuration."""

    tariff_id: str = Field(default="odisha_ht_commercial", description="Unique identifier for tariff schedule")
    name: str = Field(
        default="TPCODL HT-Commercial / Institutional (11kV/33kV)",
        description="Official name / utility category of tariff schedule",
    )
    energy_rate_inr_per_kwh: float = Field(default=8.50, description="Base commercial energy rate in INR/kWh")
    demand_charge_inr_per_kva_month: float = Field(
        default=250.0, description="Monthly fixed maximum demand charge in INR per kVA"
    )
    contract_demand_kva: float = Field(
        default=450.0, description="Sanctioned contract demand capacity in kVA"
    )
    billing_demand_percent: float = Field(
        default=80.0,
        description="Minimum billing demand percentage of contract demand (typically 80%)",
    )
    pf_penalty_threshold: float = Field(
        default=0.90,
        description="Power factor threshold below which penalty surcharge is applied (1% per 0.01 drop)",
    )
    pf_incentive_threshold: float = Field(
        default=0.95,
        description="Power factor threshold above which incentive rebate is credited (0.5% per 0.01 rise)",
    )
    tod_bands: Dict[str, ToDBand] = Field(
        default_factory=lambda: {
            "peak": ToDBand(start_hour=18, end_hour=22, multiplier=1.20, rate_inr=10.20),
            "off_peak": ToDBand(start_hour=0, end_hour=6, multiplier=0.85, rate_inr=7.225),
            "normal": ToDBand(start_hour=None, end_hour=None, multiplier=1.00, rate_inr=8.50),
        },
        description="Time-of-Day tariff pricing bands",
    )
    disclaimer: str = Field(
        default="Illustrative tariff based on OERC FY2024-25 tariff order; replace with facility's actual power supply agreement.",
        description="Regulatory / legal disclaimer",
    )

    def get_tod_band_for_hour(self, hour: int) -> Tuple[str, ToDBand]:
        """Return the band name and ToDBand applicable for an hour of day (0-23)."""
        h = int(hour) % 24
        # Check peak band
        peak = self.tod_bands.get("peak")
        if peak and peak.start_hour is not None and peak.end_hour is not None:
            if peak.start_hour <= h < peak.end_hour:
                return "peak", peak

        # Check off-peak band
        off_peak = self.tod_bands.get("off_peak")
        if off_peak and off_peak.start_hour is not None and off_peak.end_hour is not None:
            if off_peak.start_hour <= h < off_peak.end_hour:
                return "off_peak", off_peak

        # Normal band
        normal = self.tod_bands.get("normal")
        if normal is None:
            normal = ToDBand(multiplier=1.0, rate_inr=self.energy_rate_inr_per_kwh)
        return "normal", normal

    def get_rate_for_hour(self, hour: int) -> float:
        """Return the energy rate (INR/kWh) for a given hour of day (0-23)."""
        _, band = self.get_tod_band_for_hour(hour)
        return band.rate_inr

    def calculate_billing_demand(
        self, recorded_peak_kva: float, contract_demand_override: Optional[float] = None
    ) -> float:
        """Calculate monthly billing demand in kVA under Indian regulatory rules.

        Rule: Billed on max(recorded_kva, billing_demand_percent * contract_demand).
        """
        cd = float(contract_demand_override if contract_demand_override is not None else self.contract_demand_kva)
        min_demand = cd * (self.billing_demand_percent / 100.0)
        return round(max(float(recorded_peak_kva), min_demand), 2)

    def calculate_demand_charge(
        self, recorded_peak_kva: float, contract_demand_override: Optional[float] = None
    ) -> float:
        """Calculate monthly demand charge in INR based on recorded peak kVA and contract demand."""
        billing_demand = self.calculate_billing_demand(recorded_peak_kva, contract_demand_override)
        return round(billing_demand * self.demand_charge_inr_per_kva_month, 2)

    def calculate_pf_adjustment(
        self, average_pf: float, total_energy_charges_inr: float
    ) -> Tuple[float, str, float]:
        """Calculate power factor penalty surcharge or incentive rebate under Indian tariff rules.

        Rules:
          - If average_pf < 0.90: Penalty of 1.0% of total energy charges per 0.01 drop below 0.90.
          - If average_pf > 0.95: Incentive rebate of 0.5% of total energy charges per 0.01 rise above 0.95.
          - If 0.90 <= average_pf <= 0.95: No penalty or incentive.

        Returns:
            Tuple of (adjustment_amount_inr, status_label, percentage_adjustment)
            - Positive adjustment_amount_inr represents a surcharge (penalty).
            - Negative adjustment_amount_inr represents a rebate/credit (incentive).
        """
        pf = max(0.10, min(1.00, float(average_pf)))
        energy_charges = max(0.0, float(total_energy_charges_inr))

        if pf < self.pf_penalty_threshold:
            # Drop below threshold in units of 0.01 (1%)
            drop_points = round((self.pf_penalty_threshold - pf) * 100.0, 2)
            penalty_fraction = drop_points * 0.01  # 1% per 0.01 drop
            penalty_inr = round(energy_charges * penalty_fraction, 2)
            return penalty_inr, "penalty", round(penalty_fraction * 100.0, 2)

        elif pf > self.pf_incentive_threshold:
            # Rise above threshold in units of 0.01 (1%)
            rise_points = round((pf - self.pf_incentive_threshold) * 100.0, 2)
            incentive_fraction = rise_points * 0.005  # 0.5% per 0.01 rise
            incentive_inr = round(energy_charges * incentive_fraction, 2)
            return -incentive_inr, "incentive", round(incentive_fraction * 100.0, 2)

        return 0.0, "normal", 0.0

    def calculate_monthly_bill(
        self,
        total_kwh: float,
        recorded_peak_kva: float,
        average_pf: float = 0.95,
        hourly_kwh: Optional[Dict[int, float]] = None,
        contract_demand_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute complete monthly electricity bill breakdown under Indian HT commercial rules."""
        # 1. Energy Charges
        if hourly_kwh:
            energy_charges = sum(
                kwh * self.get_rate_for_hour(h) for h, kwh in hourly_kwh.items()
            )
        else:
            energy_charges = total_kwh * self.energy_rate_inr_per_kwh
        energy_charges = round(energy_charges, 2)

        # 2. Demand Charges
        cd = float(contract_demand_override if contract_demand_override is not None else self.contract_demand_kva)
        billing_demand_kva = self.calculate_billing_demand(recorded_peak_kva, cd)
        demand_charges = round(billing_demand_kva * self.demand_charge_inr_per_kva_month, 2)

        # 3. Power Factor Adjustment
        pf_adj_inr, pf_status, pf_pct = self.calculate_pf_adjustment(average_pf, energy_charges)

        # 4. Total Monthly Bill
        total_bill = round(energy_charges + demand_charges + pf_adj_inr, 2)

        # Excess demand flag
        exceeds_contract = recorded_peak_kva > cd

        return {
            "energy_charges_inr": energy_charges,
            "demand_charges_inr": demand_charges,
            "billing_demand_kva": billing_demand_kva,
            "recorded_peak_kva": recorded_peak_kva,
            "contract_demand_kva": cd,
            "exceeds_contract_demand": exceeds_contract,
            "average_power_factor": average_pf,
            "pf_adjustment_inr": pf_adj_inr,
            "pf_adjustment_status": pf_status,
            "pf_adjustment_pct": pf_pct,
            "total_bill_inr": total_bill,
            "disclaimer": self.disclaimer,
        }


def _resolve_tariff_file(tariff_id: str, custom_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Resolve path to tariff YAML configuration file."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p.resolve()

    repo_root = Path(__file__).resolve().parent.parent.parent

    # Candidate locations
    candidates = [
        repo_root / "config" / "tariffs" / f"{tariff_id}.yaml",
        repo_root / "configs" / "tariffs" / f"{tariff_id}.yaml",
        repo_root / "config" / "tariffs" / f"{tariff_id}.yml",
        repo_root / "configs" / "tariffs" / f"{tariff_id}.yml",
    ]

    for cand in candidates:
        if cand.is_file():
            return cand.resolve()

    return None


def load_tariff(
    tariff_id: str = DEFAULT_TARIFF_ID, tariff_path: Optional[Union[str, Path]] = None
) -> IndianTariff:
    """Load an IndianTariff configuration from YAML with graceful fallback.

    Args:
        tariff_id: Identifier of the tariff file (e.g. 'odisha_ht_commercial').
        tariff_path: Optional explicit file path.

    Returns:
        IndianTariff instance.
    """
    resolved_path = _resolve_tariff_file(tariff_id, tariff_path)
    if not resolved_path or not resolved_path.is_file():
        logger.warning(
            "Tariff configuration '%s' not found. Using default IndianTariff model.",
            tariff_id,
        )
        return IndianTariff(tariff_id=tariff_id)

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Parse tod_bands if present
        tod_raw = data.get("tod_bands", {})
        tod_parsed: Dict[str, ToDBand] = {}
        for b_name, b_data in tod_raw.items():
            if isinstance(b_data, dict):
                tod_parsed[b_name] = ToDBand(**b_data)

        tariff_kwargs = {
            "tariff_id": str(data.get("tariff_id", tariff_id)),
            "name": str(data.get("name", "TPCODL HT-Commercial / Institutional (11kV/33kV)")),
            "energy_rate_inr_per_kwh": float(data.get("energy_rate_inr_per_kwh", 8.50)),
            "demand_charge_inr_per_kva_month": float(data.get("demand_charge_inr_per_kva_month", 250.0)),
            "contract_demand_kva": float(data.get("contract_demand_kva", 450.0)),
            "billing_demand_percent": float(data.get("billing_demand_percent", 80.0)),
            "pf_penalty_threshold": float(data.get("pf_penalty_threshold", 0.90)),
            "pf_incentive_threshold": float(data.get("pf_incentive_threshold", 0.95)),
            "disclaimer": str(
                data.get(
                    "disclaimer",
                    "Illustrative tariff based on OERC FY2024-25 tariff order; replace with facility's actual power supply agreement.",
                )
            ),
        }
        if tod_parsed:
            tariff_kwargs["tod_bands"] = tod_parsed

        return IndianTariff(**tariff_kwargs)

    except Exception as exc:
        logger.warning("Failed to parse tariff config from '%s': %s. Returning fallback.", resolved_path, exc)
        return IndianTariff(tariff_id=tariff_id)


def list_available_tariffs() -> List[str]:
    """Return a list of available tariff schedule IDs."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    found: set[str] = set()

    for folder in [repo_root / "config" / "tariffs", repo_root / "configs" / "tariffs"]:
        if folder.is_dir():
            for p in folder.glob("*.yaml"):
                found.add(p.stem)
            for p in folder.glob("*.yml"):
                found.add(p.stem)

    if not found:
        found.add(DEFAULT_TARIFF_ID)

    return sorted(list(found))
