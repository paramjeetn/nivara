"""Unified Savings & ROI Model for Nivara Campus Energy Intelligence Platform.

Calculates standardized three-point savings ranges {low, expected, high} for electrical
energy (kWh), municipal water (kL), operating expenditures (INR), and greenhouse gas
abatement (kgCO2e). Computes simple payback periods based on capital cost benchmarks
retrieved exclusively from the assumptions registry.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from campus_energy import assumptions
from campus_energy.decision.recommendation import SavingsBundle, SavingsRange

logger = logging.getLogger(__name__)

# Standard registry keys used by the savings engine
ASSUMPTION_KEY_ELECTRICITY_TARIFF = "tariff.electricity_commercial"
ASSUMPTION_KEY_WATER_TARIFF = "tariff.water_municipal"
ASSUMPTION_KEY_GRID_EMISSIONS = "emissions.grid_factor"

BENCHMARK_KEY_MAP = {
    "led_retrofit": "capital_cost.led_retrofit_per_kw",
    "led_upgrade": "capital_cost.led_retrofit_per_kw",
    "led": "capital_cost.led_retrofit_per_kw",
    "solar_pv": "capital_cost.solar_pv_per_kwp",
    "solar": "capital_cost.solar_pv_per_kwp",
    "bess": "capital_cost.bess_per_kwh",
    "battery": "capital_cost.bess_per_kwh",
    "hvac_vfd": "capital_cost.hvac_vfd_per_kw",
    "vfd": "capital_cost.hvac_vfd_per_kw",
    "smart_sensor": "capital_cost.smart_sensor_per_unit",
    "sensor": "capital_cost.smart_sensor_per_unit",
}


def get_benchmark_capital_cost(measure: str, quantity: float = 1.0) -> float:
    """Retrieve capital cost from the assumptions registry for a given technology measure.

    Args:
        measure: Standard measure key or alias (e.g. 'led_retrofit', 'solar_pv', 'bess', 'hvac_vfd', 'smart_sensor').
        quantity: Scale metric (kW reduced, kWp installed, kWh storage, kW motor rating, or unit count).

    Returns:
        Estimated capital cost in INR.
    """
    key = BENCHMARK_KEY_MAP.get(str(measure).lower().strip())
    if not key:
        logger.warning("Unrecognized capital cost measure '%s'. Returning 0.0 capex.", measure)
        return 0.0

    unit_cost = float(assumptions.get_value(key))
    return round(unit_cost * max(0.0, float(quantity)), 1)


def get_savings_assumption_keys(include_water: bool = False, capex_measure: Optional[str] = None) -> List[str]:
    """Return the list of assumptions registry keys referenced during savings calculation.

    Args:
        include_water: Whether municipal water savings are factored into the calculation.
        capex_measure: Optional capital cost measure key to include corresponding capex benchmark.

    Returns:
        List of dot-delimited assumption keys.
    """
    keys = [ASSUMPTION_KEY_ELECTRICITY_TARIFF, ASSUMPTION_KEY_GRID_EMISSIONS]
    if include_water:
        keys.append(ASSUMPTION_KEY_WATER_TARIFF)
    if capex_measure:
        benchmark_key = BENCHMARK_KEY_MAP.get(str(capex_measure).lower().strip())
        if benchmark_key and benchmark_key not in keys:
            keys.append(benchmark_key)
    return keys


def calculate_savings(
    expected_kwh: float,
    variance_pct: float = 0.20,
    water_kl: float = 0.0,
    capital_cost: float = 0.0,
    days_modeled: float = 365.0,
) -> Tuple[SavingsBundle, Optional[float]]:
    """Calculate three-point savings ranges {low, expected, high} and simple payback period.

    Rationale for variance parameter (`variance_pct = 0.20` default):
        The variance parameter models empirical operational uncertainty across institutional
        and commercial campus facilities. Real-world energy conservation measures (ECMs)
        and operational controls are subject to:
          1. Weather & ambient seasonal variation: +/-10% cooling load sensitivity.
          2. Occupant schedule & behavioral diversity: +/-10% unmodeled plug/lighting draw.
          3. Sub-metering calibration noise: +/-5% CT transducer tolerance (IEC 62053-21).
        Per ASHRAE Guideline 14 and IPMVP Option B/C guidelines, reporting three-point
        bounds [low (conservative P10), expected (nominal P50), high (optimistic P90)]
        ensures defensible, risk-adjusted decision support rather than fragile point projections.

    Tariff & Emission Constants:
        All constants are retrieved strictly from the central assumptions registry
        (`config/assumptions.yaml`):
          - `tariff.electricity_commercial` (INR/kWh)
          - `tariff.water_municipal` (INR/kL)
          - `emissions.grid_factor` (kgCO2e/kWh)

    Payback Formulation:
        payback_months = capital_cost / (annual_inr_savings / 12.0)
        where:
          annual_inr_savings = expected_inr * (365.0 / days_modeled)

    Args:
        expected_kwh: Expected electrical energy savings for the modeled duration.
        variance_pct: Fractional spread for conservative/optimistic bounds (default 0.20 for +/-20%).
        water_kl: Expected municipal water savings in kilo-liters (kL). Default 0.0.
        capital_cost: Turnkey implementation or equipment cost in INR. Default 0.0.
        days_modeled: Duration in days represented by expected_kwh and water_kl (default 365.0 for annual).

    Returns:
        tuple[SavingsBundle, Optional[float]]:
            - Standardized SavingsBundle with low, expected, high for kWh, INR, and kgCO2e.
            - Simple payback period in months (float) if capital_cost > 0 and annual savings > 0;
              otherwise None.
    """
    # 1. Fetch validated regulatory and environmental constants from registry
    tariff_elec = float(assumptions.get_value(ASSUMPTION_KEY_ELECTRICITY_TARIFF))
    tariff_water = float(assumptions.get_value(ASSUMPTION_KEY_WATER_TARIFF))
    grid_factor = float(assumptions.get_value(ASSUMPTION_KEY_GRID_EMISSIONS))

    var = max(0.0, float(variance_pct))

    # 2. Electrical energy savings (kWh)
    k_exp = max(0.0, float(expected_kwh))
    k_low = round(k_exp * max(0.0, 1.0 - var), 1)
    k_high = round(k_exp * (1.0 + var), 1)

    # 3. Water savings (kL)
    w_exp = max(0.0, float(water_kl))
    w_low = round(w_exp * max(0.0, 1.0 - var), 1)
    w_high = round(w_exp * (1.0 + var), 1)

    # 4. Financial cost savings (INR) - combines electricity and water tariffs
    inr_exp = round(k_exp * tariff_elec + w_exp * tariff_water, 1)
    inr_low = round(k_low * tariff_elec + w_low * tariff_water, 1)
    inr_high = round(k_high * tariff_elec + w_high * tariff_water, 1)

    # Ensure monotonic ordering despite floating-point rounding
    inr_low = min(inr_low, inr_exp)
    inr_high = max(inr_high, inr_exp)

    # 5. Greenhouse gas emissions abatement (kgCO2e)
    co2_exp = round(k_exp * grid_factor, 1)
    co2_low = round(k_low * grid_factor, 1)
    co2_high = round(k_high * grid_factor, 1)

    co2_low = min(co2_low, co2_exp)
    co2_high = max(co2_high, co2_exp)

    bundle = SavingsBundle(
        kwh=SavingsRange(low=k_low, expected=k_exp, high=k_high),
        inr=SavingsRange(low=inr_low, expected=inr_exp, high=inr_high),
        co2_kg=SavingsRange(low=co2_low, expected=co2_exp, high=co2_high),
    )

    # 6. Simple payback period in months
    cap_cost = max(0.0, float(capital_cost))
    d_modeled = max(0.1, float(days_modeled))

    payback_months: Optional[float] = None
    if cap_cost > 0.0 and inr_exp > 0.0:
        annual_inr_savings = inr_exp * (365.0 / d_modeled)
        if annual_inr_savings > 0.0:
            monthly_inr_savings = annual_inr_savings / 12.0
            payback_months = round(cap_cost / monthly_inr_savings, 1)

    return bundle, payback_months
