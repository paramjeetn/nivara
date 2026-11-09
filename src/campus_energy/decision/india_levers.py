"""India-Specific Recommendation Levers for Nivara Platform.

Implements institutional tariff optimization levers reflecting how Indian facilities pay for power:
  1. Maximum Demand (kVA) Peak Shaving (contract demand compliance, demand charges, BESS)
  2. Power Factor (PF) Penalty Avoidance & APFC Retrofit (OERC/TPCODL low-PF surcharge avoidance & incentive)
  3. Time-of-Day (ToD) Tariff Load Shifting (shifting deferrable pumping / batch loads to night off-peak)
  4. Diesel Generator (DG) Audit (fuel cost, specific consumption, emissions, and outage roadmap)
"""
from __future__ import annotations

import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from campus_energy import assumptions
from campus_energy.decision.recommendation import GuardResult, Recommendation, SavingsBundle, SavingsRange
from campus_energy.decision.savings import (
    ASSUMPTION_KEY_ELECTRICITY_TARIFF,
    ASSUMPTION_KEY_GRID_EMISSIONS,
    get_benchmark_capital_cost,
    get_savings_assumption_keys,
)
from campus_energy.profiles import FacilityProfile
from campus_energy.tariffs import DEFAULT_TARIFF_ID, IndianTariff, load_tariff

logger = logging.getLogger(__name__)

# Assumption keys specific to India levers
ASSUMPTION_KEY_DEMAND_CHARGE = "tariff.demand_charge_inr_per_kva_month"
ASSUMPTION_KEY_APFC_CAPEX = "capital_cost.apfc_per_kvar"
ASSUMPTION_KEY_BESS_CAPEX = "capital_cost.bess_per_kwh"
ASSUMPTION_KEY_DIESEL_TARIFF = "tariff.diesel_inr_per_l"
ASSUMPTION_KEY_DIESEL_EMISSIONS = "emissions.diesel_kg_per_l"
ASSUMPTION_KEY_DG_SFC = "dg.specific_fuel_consumption_l_per_kwh"


def _generate_rec_id(prefix: str = "REC-IND") -> str:
    """Generate a clean, reproducible recommendation identifier."""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Maximum Demand (kVA) Peak Shaving Lever
# ─────────────────────────────────────────────────────────────────────────────

def calculate_telemetry_kva(
    telemetry_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute instantaneous/interval apparent power (kVA = kW / PF) across telemetry records.

    Args:
        telemetry_df: DataFrame with 'power_kw' and optionally 'power_factor'.

    Returns:
        DataFrame with an added 'power_kva' column.
    """
    df = telemetry_df.copy()
    if "power_kw" not in df.columns:
        df["power_kw"] = 0.0

    if "power_factor" in df.columns:
        # Clamp power factor to physical range [0.10, 1.00] to prevent zero division
        pf_series = df["power_factor"].fillna(0.85).clip(lower=0.10, upper=1.00)
    else:
        pf_series = pd.Series(0.85, index=df.index)

    df["power_kva"] = (df["power_kw"] / pf_series).round(2)
    return df


def generate_peak_demand_recommendation(
    telemetry_df: pd.DataFrame,
    profile: Optional[FacilityProfile] = None,
    tariff: Optional[IndianTariff] = None,
    contract_demand_kva: Optional[float] = None,
) -> Optional[Recommendation]:
    """Generate maximum demand (kVA) peak shaving recommendation.

    Analyzes power and power factor telemetry: kVA = kW / PF.
    Detects peak kVA demand approaching or exceeding sanctioned contract demand.
    Calculates savings from reduced kVA billing demand and sizes BESS peak shaving.

    Args:
        telemetry_df: Telemetry readings DataFrame.
        profile: Institutional FacilityProfile.
        tariff: IndianTariff instance (defaults to OERC HT Commercial).
        contract_demand_kva: Optional contract demand override.

    Returns:
        Recommendation instance or None if demand is well within baseline bounds.
    """
    if telemetry_df is None or telemetry_df.empty:
        return None

    tariff = tariff or load_tariff()
    fac_id = profile.facility_id if profile else "university"

    # Sanctioned contract demand
    cd = float(contract_demand_kva or getattr(tariff, "contract_demand_kva", 450.0))

    # Compute kVA
    df_kva = calculate_telemetry_kva(telemetry_df)

    # Determine peak recorded demand
    # If meter-level telemetry exists, aggregate by timestamp to find total campus demand
    if "timestamp" in df_kva.columns:
        meter_mask = (
            df_kva["reading_type"].eq("meter")
            if "reading_type" in df_kva.columns
            else df_kva["asset_id"].str.startswith("MTR")
        )
        if meter_mask.any():
            demand_ts = df_kva[meter_mask].groupby("timestamp")["power_kva"].sum()
        else:
            demand_ts = df_kva.groupby("timestamp")["power_kva"].sum()
        peak_recorded_kva = float(demand_ts.max()) if not demand_ts.empty else float(df_kva["power_kva"].max())
    else:
        peak_recorded_kva = float(df_kva["power_kva"].max())

    if peak_recorded_kva <= 0:
        return None

    # Billing demand threshold (80% of contract demand)
    billing_threshold_kva = cd * (tariff.billing_demand_percent / 100.0)
    recorded_billing_demand = max(peak_recorded_kva, billing_threshold_kva)

    # Trigger threshold: peak reaches or exceeds 80% of contract demand (or exceeds contract demand)
    trigger_ratio = peak_recorded_kva / cd
    if trigger_ratio < 0.80:
        # Demand is safely below billing threshold
        return None

    # Determine target peak reduction delta_kva
    # Aim to shave excess demand above 80% contract demand (or shave 10-15% of peak)
    if peak_recorded_kva > cd:
        # Exceeds contract demand: penalty risk! Shave down to 85% CD
        delta_kva = peak_recorded_kva - (0.85 * cd)
        urgency = "CRITICAL: Recorded peak exceeds sanctioned contract demand."
    else:
        # Approaching contract demand: shave 10% of peak load
        delta_kva = min(peak_recorded_kva - billing_threshold_kva, peak_recorded_kva * 0.12)
        urgency = "Recorded peak demand approaches contract demand threshold."

    delta_kva = max(5.0, round(delta_kva, 1))

    # New billing demand with peak shaving
    target_peak_kva = peak_recorded_kva - delta_kva
    target_billing_demand = max(target_peak_kva, billing_threshold_kva)
    reduced_billing_demand = recorded_billing_demand - target_billing_demand

    # Monthly & annual demand charge savings
    demand_rate = float(tariff.demand_charge_inr_per_kva_month)
    monthly_inr_savings = reduced_billing_demand * demand_rate
    annual_inr_savings = monthly_inr_savings * 12.0

    # Model BESS for 2-hour peak shaving window
    discharge_hours = 2.0
    depth_of_discharge = 0.85
    bess_kwh_required = round((delta_kva * discharge_hours) / depth_of_discharge, 1)

    # BESS capital cost from assumptions
    bess_capex_per_kwh = float(assumptions.get_value(ASSUMPTION_KEY_BESS_CAPEX, 18000.0))
    bess_total_capex = round(bess_kwh_required * bess_capex_per_kwh, 1)

    # Payback period in months
    payback_months = (
        round((bess_total_capex / annual_inr_savings) * 12.0, 1) if annual_inr_savings > 0 else None
    )

    # SavingsBundle
    bundle = SavingsBundle(
        kwh=SavingsRange(low=0.0, expected=0.0, high=0.0),  # Demand shaving saves billing kVA, not net energy
        inr=SavingsRange(
            low=round(annual_inr_savings * 0.85, 1),
            expected=round(annual_inr_savings, 1),
            high=round(annual_inr_savings * 1.15, 1),
        ),
        co2_kg=SavingsRange(low=0.0, expected=0.0, high=0.0),
    )

    action_text = (
        f"Implement {delta_kva:.1f} kVA maximum demand shaving via BESS dispatch "
        f"({bess_kwh_required:.0f} kWh capacity) or non-critical load staggering during peak windows. "
        f"{urgency} Reduces monthly billing demand from {recorded_billing_demand:.1f} kVA to "
        f"{target_billing_demand:.1f} kVA, avoiding ₹{annual_inr_savings:,.0f}/yr in demand charges."
    )

    assumption_refs = [
        "tariff.demand_charge_inr_per_kva_month",
        ASSUMPTION_KEY_BESS_CAPEX,
    ]

    return Recommendation(
        id=_generate_rec_id("REC-IND-DMD"),
        facility_id=fac_id,
        type="peak_demand_shaving",
        target="MTR-MAIN",
        action_text=action_text,
        trigger=f"peak_demand_kva:{peak_recorded_kva:.1f}_vs_cd:{cd:.1f}",
        savings=bundle,
        cost=bess_total_capex,
        payback_months=payback_months,
        assumption_refs=assumption_refs,
        owner_role="operations",
        confidence=0.90,
        is_estimate=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Power Factor (PF) Penalty Avoidance & APFC Retrofit Lever
# ─────────────────────────────────────────────────────────────────────────────

def calculate_apfc_requirements(
    active_power_kw: float,
    current_pf: float,
    target_pf: float = 0.98,
) -> float:
    """Calculate required capacitor bank rating (kVAR) to raise power factor to target.

    Formula:
        kVAR = P * (tan(phi_1) - tan(phi_2))
        where tan(phi) = sqrt(1 - PF^2) / PF

    Args:
        active_power_kw: Average or operating active load in kW.
        current_pf: Current observed power factor.
        target_pf: Target corrected power factor (default 0.98).

    Returns:
        Required capacitor rating in kVAR.
    """
    p = max(0.0, float(active_power_kw))
    pf1 = max(0.10, min(0.999, float(current_pf)))
    pf2 = max(0.10, min(0.999, float(target_pf)))

    if pf1 >= pf2:
        return 0.0

    tan_phi1 = math.sqrt(1.0 - pf1**2) / pf1
    tan_phi2 = math.sqrt(1.0 - pf2**2) / pf2
    kvar = p * (tan_phi1 - tan_phi2)

    # Standard commercial sizing: round to nearest 5 kVAR, minimum 5 kVAR
    kvar_rounded = max(5.0, round(kvar / 5.0) * 5.0)
    return kvar_rounded


def generate_power_factor_recommendation(
    telemetry_df: pd.DataFrame,
    profile: Optional[FacilityProfile] = None,
    tariff: Optional[IndianTariff] = None,
    target_pf: float = 0.98,
) -> Optional[Recommendation]:
    """Generate Power Factor (PF) penalty avoidance and APFC retrofit recommendation.

    Scans feeder and meter telemetry for average PF or intervals where PF < 0.90.
    Calculates penalty surcharge under Indian tariff rules and potential incentive bonus.
    Sizes APFC capacitor bank and calculates turnkey capex and payback.

    Args:
        telemetry_df: Telemetry readings DataFrame.
        profile: Institutional FacilityProfile.
        tariff: IndianTariff instance (defaults to OERC HT Commercial).
        target_pf: Desired target power factor (default 0.98).

    Returns:
        Recommendation instance or None if PF is compliant (>= 0.90).
    """
    if telemetry_df is None or telemetry_df.empty:
        return None

    tariff = tariff or load_tariff()
    fac_id = profile.facility_id if profile else "university"

    # Scan power factor
    if "power_factor" not in telemetry_df.columns:
        return None

    valid_pf = telemetry_df["power_factor"].dropna()
    valid_pf = valid_pf[(valid_pf >= 0.10) & (valid_pf <= 1.00)]
    if valid_pf.empty:
        return None

    avg_pf = float(valid_pf.mean())

    # Evaluate against regulatory penalty threshold (0.90)
    if avg_pf >= tariff.pf_penalty_threshold:
        return None

    # Estimate active load and monthly energy consumption
    active_power_mean = float(telemetry_df["power_kw"].dropna().mean()) if "power_kw" in telemetry_df.columns else 50.0
    active_power_mean = max(10.0, active_power_mean)

    # Duration and monthly scaling
    if "timestamp" in telemetry_df.columns:
        ts = pd.to_datetime(telemetry_df["timestamp"], errors="coerce").dropna()
        if len(ts) >= 2:
            duration_days = max(1.0, (ts.max() - ts.min()).total_seconds() / 86400.0)
        else:
            duration_days = 30.0
    else:
        duration_days = 30.0

    if "interval_energy_kwh" in telemetry_df.columns:
        total_kwh_sample = float(telemetry_df["interval_energy_kwh"].sum())
        monthly_kwh = total_kwh_sample * (30.0 / duration_days)
    else:
        # Fallback based on average active power
        monthly_kwh = active_power_mean * 24.0 * 30.0

    energy_rate = float(tariff.energy_rate_inr_per_kwh)
    monthly_energy_charges = monthly_kwh * energy_rate

    # Calculate current penalty surcharge
    penalty_inr, _, penalty_pct = tariff.calculate_pf_adjustment(avg_pf, monthly_energy_charges)
    annual_penalty_avoided = penalty_inr * 12.0

    # Calculate potential incentive bonus at target_pf (e.g. 0.98)
    incentive_inr, _, incentive_pct = tariff.calculate_pf_adjustment(target_pf, monthly_energy_charges)
    # Note: calculate_pf_adjustment returns negative value for incentive rebate
    monthly_incentive_gain = abs(incentive_inr)
    annual_incentive_gain = monthly_incentive_gain * 12.0

    total_annual_financial_gain = annual_penalty_avoided + annual_incentive_gain

    # Size required APFC capacitor bank in kVAR
    kvar_required = calculate_telemetry_kva(telemetry_df)
    # Use average campus active load for capacitor sizing
    kvar_rating = calculate_apfc_requirements(active_power_mean, avg_pf, target_pf)

    # Capex from assumptions
    apfc_capex_per_kvar = float(assumptions.get_value(ASSUMPTION_KEY_APFC_CAPEX, 1200.0))
    total_capex = round(kvar_rating * apfc_capex_per_kvar, 1)

    # Payback
    payback_months = (
        round((total_capex / total_annual_financial_gain) * 12.0, 1) if total_annual_financial_gain > 0 else None
    )

    # SavingsBundle
    bundle = SavingsBundle(
        kwh=SavingsRange(low=0.0, expected=0.0, high=0.0),
        inr=SavingsRange(
            low=round(total_annual_financial_gain * 0.90, 1),
            expected=round(total_annual_financial_gain, 1),
            high=round(total_annual_financial_gain * 1.10, 1),
        ),
        co2_kg=SavingsRange(low=0.0, expected=0.0, high=0.0),
    )

    action_text = (
        f"Retrofit {kvar_rating:.0f} kVAR Automatic Power Factor Correction (APFC) capacitor bank "
        f"to raise average power factor from {avg_pf:.3f} to {target_pf:.2f}. "
        f"Eliminates TPCODL low-PF penalty surcharge ({penalty_pct:.1f}% on monthly energy charges = ₹{annual_penalty_avoided:,.0f}/yr) "
        f"and captures {incentive_pct:.1f}% high-PF incentive bonus (₹{annual_incentive_gain:,.0f}/yr)."
    )

    assumption_refs = [
        "tariff.electricity_commercial",
        "tariff.pf_penalty_threshold",
        "tariff.pf_incentive_threshold",
        ASSUMPTION_KEY_APFC_CAPEX,
    ]

    return Recommendation(
        id=_generate_rec_id("REC-IND-APFC"),
        facility_id=fac_id,
        type="power_factor_correction",
        target="Main Substation APFC Panel",
        action_text=action_text,
        trigger=f"average_power_factor:{avg_pf:.3f}_below_threshold:{tariff.pf_penalty_threshold:.2f}",
        savings=bundle,
        cost=total_capex,
        payback_months=payback_months,
        assumption_refs=assumption_refs,
        owner_role="operations",
        confidence=0.94,
        is_estimate=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Time-of-Day (ToD) Tariff Load Shifting Lever
# ─────────────────────────────────────────────────────────────────────────────

DEFERRABLE_ASSET_KEYWORDS = [
    "pump",
    "pm",
    "water",
    "transfer",
    "laundry",
    "washer",
    "heater",
    "geyser",
    "calorifier",
]


def _is_deferrable_load(asset_id: str, asset_name: str = "") -> bool:
    """Return True if asset represents a deferrable institutional load."""
    txt = f"{asset_id} {asset_name}".lower()
    return any(k in txt for k in DEFERRABLE_ASSET_KEYWORDS)


def generate_tod_load_shifting_recommendations(
    telemetry_df: pd.DataFrame,
    profile: Optional[FacilityProfile] = None,
    tariff: Optional[IndianTariff] = None,
) -> List[Recommendation]:
    """Generate Time-of-Day (ToD) tariff load shifting recommendations.

    Identifies deferrable loads (water transfer pumps like EQ-UTL-P1-PM, batch laundry, water heaters)
    operating during peak hours (18:00–22:00).
    Recommends shifting pump operation to off-peak night hours (01:00–05:00).
    Calculates ToD differential savings: (peak_rate - off_peak_rate) * shifted_kwh.

    Args:
        telemetry_df: Telemetry readings DataFrame.
        profile: Institutional FacilityProfile.
        tariff: IndianTariff instance.

    Returns:
        List of unverified Recommendation instances.
    """
    if telemetry_df is None or telemetry_df.empty:
        return []

    tariff = tariff or load_tariff()
    fac_id = profile.facility_id if profile else "university"

    # Peak and off-peak ToD rates
    peak_rate = tariff.tod_bands.get("peak", tariff.tod_bands.get("normal")).rate_inr
    off_peak_rate = tariff.tod_bands.get("off_peak", tariff.tod_bands.get("normal")).rate_inr
    rate_differential = max(0.0, peak_rate - off_peak_rate)

    if "timestamp" not in telemetry_df.columns:
        return []

    df = telemetry_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"])
    if df.empty:
        return []

    df["hour"] = df["ts"].dt.hour

    # Peak window: 18:00 to 22:00 (hours 18, 19, 20, 21)
    peak_band = tariff.tod_bands.get("peak")
    peak_start = peak_band.start_hour if peak_band and peak_band.start_hour is not None else 18
    peak_end = peak_band.end_hour if peak_band and peak_band.end_hour is not None else 22

    # Calculate telemetry duration in days
    duration_days = max(1.0, (df["ts"].max() - df["ts"].min()).total_seconds() / 86400.0)

    recommendations: List[Recommendation] = []

    # Identify deferrable assets present in telemetry
    unique_assets = df["asset_id"].unique()
    deferrable_assets = [a for a in unique_assets if _is_deferrable_load(str(a))]

    # If no deferrable assets match by keyword, but profile contains utility pump rooms, check for them
    if not deferrable_assets and profile:
        for pl in profile.protected_loads:
            if _is_deferrable_load(pl.asset_id, pl.asset_name):
                deferrable_assets.append(pl.asset_id)

    for asset_id in deferrable_assets:
        asset_df = df[df["asset_id"] == asset_id]
        if asset_df.empty:
            continue

        # Filter for peak window operation
        peak_records = asset_df[(asset_df["hour"] >= peak_start) & (asset_df["hour"] < peak_end)]
        if peak_records.empty:
            continue

        if "interval_energy_kwh" in peak_records.columns:
            peak_kwh_total = float(peak_records["interval_energy_kwh"].sum())
        elif "power_kw" in peak_records.columns:
            # Estimate kWh from power_kw assuming 15-min intervals
            peak_kwh_total = float(peak_records["power_kw"].sum() * 0.25)
        else:
            continue

        if peak_kwh_total <= 0.5:
            continue

        # Daily and annual energy shifted
        daily_shifted_kwh = peak_kwh_total / duration_days
        annual_shifted_kwh = daily_shifted_kwh * 365.0

        # Cost savings from ToD differential
        annual_inr_savings = annual_shifted_kwh * rate_differential

        # SavingsBundle
        bundle = SavingsBundle(
            kwh=SavingsRange(low=0.0, expected=0.0, high=0.0),  # Net energy shifted, not saved
            inr=SavingsRange(
                low=round(annual_inr_savings * 0.85, 1),
                expected=round(annual_inr_savings, 1),
                high=round(annual_inr_savings * 1.15, 1),
            ),
            co2_kg=SavingsRange(low=0.0, expected=0.0, high=0.0),
        )

        action_text = (
            f"Reschedule {asset_id} operating schedule from evening peak ToD window "
            f"({peak_start:02d}:00–{peak_end:02d}:00 at ₹{peak_rate:.2f}/kWh) to night off-peak window "
            f"(01:00–05:00 at ₹{off_peak_rate:.2f}/kWh). Shifting {annual_shifted_kwh:,.0f} kWh/year "
            f"captures ₹{rate_differential:.3f}/kWh ToD arbitrage saving ₹{annual_inr_savings:,.0f}/yr. "
            f"Service Guard verification: Ensure overhead water storage tank minimum level is maintained."
        )

        assumption_refs = [
            "tariff.electricity_commercial",
            "tariff.tod_peak",
            "tariff.tod_off_peak",
        ]

        rec = Recommendation(
            id=_generate_rec_id("REC-IND-TOD"),
            facility_id=fac_id,
            type="tod_load_shifting",
            target=str(asset_id),
            action_text=action_text,
            trigger=f"tod_peak_operation:{asset_id}(peak_kwh_day={daily_shifted_kwh:.1f})",
            savings=bundle,
            cost=0.0,  # Zero capex: software / BMS schedule timer adjustment
            payback_months=0.0,
            assumption_refs=assumption_refs,
            owner_role="operations",
            confidence=0.88,
            is_estimate=True,
        )
        recommendations.append(rec)

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# 4. Diesel Generator (DG) Audit & Outage Roadmap Lever
# ─────────────────────────────────────────────────────────────────────────────

def audit_diesel_generators(
    profile: Optional[FacilityProfile] = None,
    telemetry_df: Optional[pd.DataFrame] = None,
    tariff: Optional[IndianTariff] = None,
) -> Dict[str, Any]:
    """Audit facility diesel generator assets, operational costs, and emissions factors.

    Reports diesel fuel procurement cost (₹95/L) and direct emissions factor (~2.68 kg CO2/L),
    calculates generation cost per kWh (~₹28.50/kWh) compared to grid power (₹8.50/kWh),
    and flags real-time grid outage simulation as roadmap.

    Args:
        profile: Institutional FacilityProfile.
        telemetry_df: Optional telemetry DataFrame.
        tariff: IndianTariff instance.

    Returns:
        Audit report dictionary.
    """
    tariff = tariff or load_tariff()
    fac_id = profile.facility_id if profile else "campus"

    # Assumptions lookup
    diesel_price_inr_per_l = float(assumptions.get_value(ASSUMPTION_KEY_DIESEL_TARIFF, 95.0))
    diesel_emissions_factor = float(assumptions.get_value(ASSUMPTION_KEY_DIESEL_EMISSIONS, 2.68))
    sfc_l_per_kwh = float(assumptions.get_value(ASSUMPTION_KEY_DG_SFC, 0.30))

    # Derived electricity generation metrics
    dg_cost_per_kwh = round(sfc_l_per_kwh * diesel_price_inr_per_l, 2)
    dg_emissions_per_kwh = round(sfc_l_per_kwh * diesel_emissions_factor, 3)

    grid_cost_per_kwh = tariff.energy_rate_inr_per_kwh
    grid_emissions_factor = float(assumptions.get_value(ASSUMPTION_KEY_GRID_EMISSIONS, 0.82))

    cost_multiplier_vs_grid = round(dg_cost_per_kwh / grid_cost_per_kwh, 2)

    # Detect DG assets from profile and telemetry
    dg_assets: List[str] = []
    if profile:
        for pl in profile.protected_loads:
            if any(k in f"{pl.asset_id} {pl.asset_name}".lower() for k in ["dg", "generator", "diesel"]):
                dg_assets.append(pl.asset_id)

    if telemetry_df is not None and not telemetry_df.empty:
        for aid in telemetry_df["asset_id"].dropna().unique():
            aid_str = str(aid)
            if any(k in aid_str.lower() for k in ["dg", "generator", "gen_set"]) and aid_str not in dg_assets:
                dg_assets.append(aid_str)

    has_dg = len(dg_assets) > 0

    return {
        "facility_id": fac_id,
        "has_dg_assets": has_dg,
        "dg_assets_detected": dg_assets,
        "diesel_fuel_price_inr_per_l": diesel_price_inr_per_l,
        "direct_emissions_kg_per_l": diesel_emissions_factor,
        "specific_fuel_consumption_l_per_kwh": sfc_l_per_kwh,
        "dg_generation_cost_inr_per_kwh": dg_cost_per_kwh,
        "grid_cost_inr_per_kwh": grid_cost_per_kwh,
        "cost_multiplier_vs_grid": cost_multiplier_vs_grid,
        "dg_emissions_kg_per_kwh": dg_emissions_per_kwh,
        "grid_emissions_kg_per_kwh": grid_emissions_factor,
        "roadmap_feature": (
            "Real-time grid outage detection, automatic transfer switch (ATS) telemetry logging, "
            "fuel tank ultrasonic sensor integration, and hybrid solar-BESS-diesel microgrid dispatch "
            "are currently on the Nivara development roadmap."
        ),
    }


def generate_dg_audit_recommendation(
    profile: Optional[FacilityProfile] = None,
    telemetry_df: Optional[pd.DataFrame] = None,
    tariff: Optional[IndianTariff] = None,
) -> Optional[Recommendation]:
    """Generate DG operational audit recommendation if DG assets exist."""
    audit = audit_diesel_generators(profile, telemetry_df, tariff)
    if not audit["has_dg_assets"]:
        return None

    fac_id = profile.facility_id if profile else "campus"
    assets_str = ", ".join(audit["dg_assets_detected"])

    action_text = (
        f"Institutional Diesel Generator Audit ({assets_str}): DG power generation cost is "
        f"₹{audit['dg_generation_cost_inr_per_kwh']:.2f}/kWh ({audit['cost_multiplier_vs_grid']}x grid tariff) "
        f"with direct diesel fuel emissions of {audit['direct_emissions_kg_per_l']} kg CO2/L. "
        f"Recommendation: Restrict DG operation strictly to life-critical emergency outages and utilize "
        f"solar-BESS for routine peak shaving. (Roadmap: Real-time ATS outage simulation & hybrid dispatch)."
    )

    bundle = SavingsBundle(
        kwh=SavingsRange(low=0.0, expected=0.0, high=0.0),
        inr=SavingsRange(low=0.0, expected=0.0, high=0.0),
        co2_kg=SavingsRange(low=0.0, expected=0.0, high=0.0),
    )

    assumption_refs = [
        ASSUMPTION_KEY_DIESEL_TARIFF,
        ASSUMPTION_KEY_DIESEL_EMISSIONS,
        ASSUMPTION_KEY_DG_SFC,
        ASSUMPTION_KEY_ELECTRICITY_TARIFF,
    ]

    return Recommendation(
        id=_generate_rec_id("REC-IND-DGA"),
        facility_id=fac_id,
        type="diesel_generator_audit",
        target=audit["dg_assets_detected"][0] if audit["dg_assets_detected"] else "DG Plant",
        action_text=action_text,
        trigger="dg_asset_detected_compliance_audit",
        savings=bundle,
        cost=0.0,
        payback_months=None,
        assumption_refs=assumption_refs,
        owner_role="operations",
        confidence=0.92,
        is_estimate=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Master India Tariff Recommendation Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_india_tariff_recommendations(
    telemetry_df: Optional[pd.DataFrame] = None,
    profile: Optional[FacilityProfile] = None,
    tariff: Optional[IndianTariff] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[Recommendation]:
    """Synthesize all India-specific tariff optimization recommendations.

    Covers:
      1. Maximum Demand (kVA) Peak Shaving
      2. Power Factor (PF) Penalty Avoidance & APFC Retrofit
      3. Time-of-Day (ToD) Load Shifting for deferrable pumps/heating
      4. Diesel Generator (DG) operational compliance audit

    Args:
        telemetry_df: Facility telemetry readings DataFrame.
        profile: Institutional FacilityProfile.
        tariff: IndianTariff instance (loads default if None).
        context: Optional dictionary of operational parameters.

    Returns:
        List of unverified Recommendation instances (must be passed through Service-Impact Guard).
    """
    recommendations: List[Recommendation] = []
    tariff = tariff or load_tariff()

    if telemetry_df is None or telemetry_df.empty:
        # If no telemetry passed, check if DG audit can still be generated from profile
        dg_rec = generate_dg_audit_recommendation(profile, None, tariff)
        if dg_rec:
            recommendations.append(dg_rec)
        return recommendations

    # 1. Maximum Demand Peak Shaving
    try:
        rec_demand = generate_peak_demand_recommendation(telemetry_df, profile, tariff)
        if rec_demand:
            recommendations.append(rec_demand)
    except Exception as e:
        logger.warning("Error generating peak demand recommendation: %s", e)

    # 2. Power Factor Correction & APFC Retrofit
    try:
        rec_pf = generate_power_factor_recommendation(telemetry_df, profile, tariff)
        if rec_pf:
            recommendations.append(rec_pf)
    except Exception as e:
        logger.warning("Error generating power factor recommendation: %s", e)

    # 3. Time-of-Day (ToD) Load Shifting
    try:
        rec_tod = generate_tod_load_shifting_recommendations(telemetry_df, profile, tariff)
        recommendations.extend(rec_tod)
    except Exception as e:
        logger.warning("Error generating ToD load shifting recommendations: %s", e)

    # 4. Diesel Generator Audit
    try:
        rec_dg = generate_dg_audit_recommendation(profile, telemetry_df, tariff)
        if rec_dg:
            recommendations.append(rec_dg)
    except Exception as e:
        logger.warning("Error generating DG audit recommendation: %s", e)

    return recommendations
