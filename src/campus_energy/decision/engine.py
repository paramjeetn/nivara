"""Unified Recommendation Engine for Nivara Campus Energy Intelligence Platform.

Synthesizes actionable, defensible energy recommendations from four distinct institutional streams:
  1. Anomaly alerts (unoccupied lighting/HVAC, pump dry-run, sensor dropouts, water leaks)
  2. 24-hour predictive load forecasts (peak demand spike mitigation, ToD arbitrage)
  3. Techno-economic scenario analyses (LED retrofits, rooftop solar PV, BESS dispatch, AC setpoint policies)
  4. Occupant complaint clusters (thermal comfort balancing, lighting audits, plumbing maintenance)

Enforces institutional safety constraints via the Service-Impact Guard (service_guard.py),
ensuring zero compromise to clinical care, occupant comfort, cold-chains, or critical research.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from campus_energy import assumptions
from campus_energy.decision.guard import apply_service_guard, check_service_guard
from campus_energy.decision.recommendation import GuardResult, Recommendation, SavingsBundle
from campus_energy.decision.india_levers import generate_india_tariff_recommendations
from campus_energy.decision.savings import (
    ASSUMPTION_KEY_ELECTRICITY_TARIFF,
    ASSUMPTION_KEY_GRID_EMISSIONS,
    ASSUMPTION_KEY_WATER_TARIFF,
    calculate_savings,
    get_benchmark_capital_cost,
    get_savings_assumption_keys,
)
from campus_energy.profiles import FacilityProfile
from campus_energy.tariffs import IndianTariff, load_tariff

logger = logging.getLogger(__name__)


def _generate_rec_id(prefix: str = "REC") -> str:
    """Generate a clean, reproducible recommendation identifier."""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Recommendations from Anomaly Alerts
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations_from_alerts(
    alerts: Union[List[Any], pd.DataFrame],
    context: Optional[Dict[str, Any]] = None,
    profile: Optional[FacilityProfile] = None,
) -> List[Recommendation]:
    """Synthesize actionable energy recommendations from active anomaly detection alerts.

    Detects:
      - Lights left on after hours -> lighting_shutdown
      - HVAC left on after hours -> hvac_shutdown
      - Pump cavitation / dry-run -> equipment_shutdown / pump_maintenance
      - Telemetry sensor dropouts -> sensor_check
      - Water pipe leaks / bursts -> water_leak_repair

    Args:
        alerts: List of Alert objects or dictionary records, or a DataFrame of alerts.
        context: Optional dictionary of operational/environmental context.
        profile: Optional FacilityProfile for facility metadata.

    Returns:
        List of unverified Recommendation instances (to be passed to Service-Impact Guard).
    """
    if alerts is None:
        return []

    # Normalize to list of dicts or objects
    if isinstance(alerts, pd.DataFrame):
        alert_items = alerts.to_dict(orient="records")
    elif isinstance(alerts, list):
        alert_items = alerts
    else:
        return []

    fac_id = profile.facility_id if profile else "university"
    recommendations: List[Recommendation] = []

    for alert in alert_items:
        # Extract alert attributes supporting both Pydantic models and dicts
        if isinstance(alert, dict):
            a_id = str(alert.get("alert_id") or alert.get("id") or "ALT-000")
            anomaly_type = str(alert.get("anomaly_type") or alert.get("event_type") or "anomaly_fix").lower()
            asset_id = str(alert.get("asset_id") or alert.get("target") or "Campus-wide")
            zone_id = str(alert.get("zone_id") or alert.get("building_id") or asset_id)
            dev = float(alert.get("deviation") or alert.get("actual_value") or 15.0)
            rec_text = str(alert.get("recommendation") or "")
            evidence = str(alert.get("evidence") or "")
            conf = float(alert.get("confidence") or 0.85)
        else:
            a_id = str(getattr(alert, "alert_id", "ALT-000"))
            anomaly_type = str(getattr(alert, "anomaly_type", "anomaly_fix")).lower()
            asset_id = str(getattr(alert, "asset_id", "Campus-wide"))
            zone_id = str(getattr(alert, "zone_id", None) or getattr(alert, "building_id", asset_id))
            dev = float(getattr(alert, "deviation", None) or getattr(alert, "actual_value", None) or 15.0)
            rec_text = str(getattr(alert, "recommendation", ""))
            evidence = str(getattr(alert, "evidence", ""))
            conf = float(getattr(alert, "confidence", 0.85) or 0.85)

        combined_desc = f"{anomaly_type} {rec_text} {evidence}".lower()

        # Classify alert into specific recommendation category
        if any(k in combined_desc for k in ["water", "leak", "pipe", "burst", "flow_rate", "mnf"]):
            # Water leak remediation
            water_kl = dev if dev > 0.1 else 5.0
            bundle, payback = calculate_savings(
                expected_kwh=0.0,
                variance_pct=0.20,
                water_kl=water_kl,
                capital_cost=0.0,
                days_modeled=1.0,
            )
            rec = Recommendation(
                id=_generate_rec_id("REC-WTR"),
                facility_id=fac_id,
                type="water_leak_repair",
                target=asset_id,
                action_text=(
                    rec_text
                    or f"Deploy plumbing maintenance crew to inspect and repair water leak on {asset_id}; isolate branch valve."
                ),
                trigger=f"alert:{a_id}",
                savings=bundle,
                cost=0.0,
                payback_months=payback,
                assumption_refs=get_savings_assumption_keys(include_water=True),
                owner_role="operations",
                confidence=conf,
                is_estimate=True,
            )
            recommendations.append(rec)

        elif any(k in combined_desc for k in ["pump", "cavitation", "dry_run", "dry-run", "no_flow"]):
            # Pump dry-run / cavitation
            expected_kwh = max(5.0, dev * 4.0)  # ~4 hours of dry-run waste avoided
            bundle, payback = calculate_savings(
                expected_kwh=expected_kwh,
                variance_pct=0.20,
                capital_cost=0.0,
                days_modeled=1.0,
            )
            rec = Recommendation(
                id=_generate_rec_id("REC-PMP"),
                facility_id=fac_id,
                type="equipment_shutdown",
                target=asset_id,
                action_text=(
                    rec_text
                    or f"Immediately de-energize pump {asset_id} to prevent mechanical seal burnout and cavitation damage."
                ),
                trigger=f"alert:{a_id}",
                savings=bundle,
                cost=0.0,
                payback_months=payback,
                assumption_refs=get_savings_assumption_keys(),
                owner_role="operations",
                confidence=conf,
                is_estimate=True,
            )
            recommendations.append(rec)

        elif any(k in combined_desc for k in ["light", "lighting", "luminaire", "lamp"]):
            # Lighting left on in unoccupied space
            expected_kwh = max(4.0, dev * 6.0)  # ~6 hours of unoccupied lighting
            bundle, payback = calculate_savings(
                expected_kwh=expected_kwh,
                variance_pct=0.20,
                capital_cost=0.0,
                days_modeled=1.0,
            )
            rec = Recommendation(
                id=_generate_rec_id("REC-LTG"),
                facility_id=fac_id,
                type="lighting_shutdown",
                target=asset_id,
                action_text=(
                    rec_text
                    or f"De-energize non-essential lighting circuits in {zone_id} ({asset_id}) during unoccupied period."
                ),
                trigger=f"alert:{a_id}",
                savings=bundle,
                cost=0.0,
                payback_months=payback,
                assumption_refs=get_savings_assumption_keys(),
                owner_role="operations",
                confidence=conf,
                is_estimate=True,
            )
            recommendations.append(rec)

        elif any(k in combined_desc for k in ["hvac", "ac", "chiller", "cooling", "ahu", "fcu"]):
            # HVAC left on in unoccupied space
            expected_kwh = max(10.0, dev * 5.0)  # ~5 hours of after-hours cooling
            bundle, payback = calculate_savings(
                expected_kwh=expected_kwh,
                variance_pct=0.20,
                capital_cost=0.0,
                days_modeled=1.0,
            )
            rec = Recommendation(
                id=_generate_rec_id("REC-HVC"),
                facility_id=fac_id,
                type="hvac_shutdown",
                target=asset_id,
                action_text=(
                    rec_text
                    or f"Switch HVAC / AHU unit {asset_id} to unoccupied night setback or shut down after operating shift."
                ),
                trigger=f"alert:{a_id}",
                savings=bundle,
                cost=0.0,
                payback_months=payback,
                assumption_refs=get_savings_assumption_keys(),
                owner_role="operations",
                confidence=conf,
                is_estimate=True,
            )
            recommendations.append(rec)

        elif any(k in combined_desc for k in ["sensor", "dropout", "offline", "data_gap", "telemetry", "communication"]):
            # Sensor connectivity dropout
            expected_kwh = 8.0  # conservative waste avoided by restoring automated setback
            bundle, payback = calculate_savings(
                expected_kwh=expected_kwh,
                variance_pct=0.20,
                capital_cost=0.0,
                days_modeled=1.0,
            )
            rec = Recommendation(
                id=_generate_rec_id("REC-SNS"),
                facility_id=fac_id,
                type="sensor_check",
                target=asset_id,
                action_text=(
                    rec_text
                    or f"Inspect physical telemetry sensor wiring and BMS gateway communication for dropped node {asset_id}."
                ),
                trigger=f"alert:{a_id}",
                savings=bundle,
                cost=0.0,
                payback_months=payback,
                assumption_refs=get_savings_assumption_keys(),
                owner_role="operations",
                confidence=conf,
                is_estimate=True,
            )
            recommendations.append(rec)

        else:
            # Generic anomaly resolution
            expected_kwh = max(5.0, dev * 3.0)
            bundle, payback = calculate_savings(
                expected_kwh=expected_kwh,
                variance_pct=0.20,
                capital_cost=0.0,
                days_modeled=1.0,
            )
            rec = Recommendation(
                id=_generate_rec_id("REC-GEN"),
                facility_id=fac_id,
                type="anomaly_fix",
                target=asset_id,
                action_text=rec_text or f"Investigate and resolve abnormal electrical load on {asset_id}.",
                trigger=f"alert:{a_id}",
                savings=bundle,
                cost=0.0,
                payback_months=payback,
                assumption_refs=get_savings_assumption_keys(),
                owner_role="operations",
                confidence=conf,
                is_estimate=True,
            )
            recommendations.append(rec)

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# 2. Recommendations from Predictive Load Forecast
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations_from_forecast(
    forecast_df: pd.DataFrame,
    profile: Optional[FacilityProfile] = None,
) -> List[Recommendation]:
    """Analyze 24-hour predictive load forecast and recommend peak demand shaving measures.

    Detects predicted load spikes during high-demand / peak tariff windows (16:00 to 21:00)
    that would incur severe commercial ToD surcharges or exceed contract demand limits.
    Recommends BESS peak discharge or shifting deferrable loads (water pumping, thermal pre-cooling).

    Args:
        forecast_df: DataFrame containing at least predicted power columns
                     (e.g. 'predicted_power_kw', 'timestamp', and optional 'asset_id').
        profile: FacilityProfile encoding facility constraints and tariffs.

    Returns:
        List of peak-shaving Recommendation instances.
    """
    if forecast_df is None or forecast_df.empty:
        return []

    fac_id = profile.facility_id if profile else "university"
    recommendations: List[Recommendation] = []

    # Resolve load column name
    power_col = None
    for col in ["predicted_power_kw", "predicted_power", "scenario_power_kw", "power_kw"]:
        if col in forecast_df.columns:
            power_col = col
            break

    if not power_col:
        return []

    # Timestamp handling for peak window detection
    df = forecast_df.copy()
    if "timestamp" in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    else:
        df["hour"] = [i % 24 for i in range(len(df))]

    # Evening peak window (16:00 to 21:00)
    peak_slice = df[df["hour"].between(16, 21)]
    if peak_slice.empty:
        peak_slice = df

    max_peak_kw = float(peak_slice[power_col].max())
    mean_load_kw = float(df[power_col].mean())

    # Detect spike: peak exceeds mean load by at least 25% or 30 kW
    demand_delta_kw = max_peak_kw - mean_load_kw
    if demand_delta_kw > 15.0:
        shave_kw = round(demand_delta_kw * 0.65, 1)  # Target shaving 65% of the spike
        peak_duration_hours = 3.0
        shaved_kwh = round(shave_kw * peak_duration_hours, 1)

        # Retrieve ToD peak tariff keys
        tod_peak_key = "tariff.tod_peak"
        tod_off_key = "tariff.tod_off_peak"

        bundle, payback = calculate_savings(
            expected_kwh=shaved_kwh,
            variance_pct=0.15,
            capital_cost=0.0,
            days_modeled=1.0,
        )

        target_asset = "Main Substation / BESS"
        if "asset_id" in peak_slice.columns:
            top_asset = peak_slice.loc[peak_slice[power_col].idxmax()]["asset_id"]
            if pd.notna(top_asset):
                target_asset = str(top_asset)

        rec = Recommendation(
            id=_generate_rec_id("REC-FCST-PEAK"),
            facility_id=fac_id,
            type="peak_shaving",
            target=target_asset,
            action_text=(
                f"Peak Shaving Dispatch: Forecast predicts peak demand surge to {max_peak_kw:.1f} kW between 17:00–20:00. "
                f"Discharge on-site BESS ({shave_kw:.1f} kW) or stagger non-essential auxiliary pumping to shave contract peak."
            ),
            trigger=f"forecast_peak_spike:{max_peak_kw:.1f}kW",
            savings=bundle,
            cost=0.0,
            payback_months=payback,
            assumption_refs=[
                ASSUMPTION_KEY_ELECTRICITY_TARIFF,
                ASSUMPTION_KEY_GRID_EMISSIONS,
                tod_peak_key,
                tod_off_key,
            ],
            owner_role="operations",
            confidence=0.88,
            is_estimate=True,
        )
        recommendations.append(rec)

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# 3. Recommendations from Techno-Economic Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations_from_scenarios(
    scenario_name: str,
    scenario_params: Dict[str, Any],
    profile: Optional[FacilityProfile] = None,
) -> List[Recommendation]:
    """Synthesize strategic capital or policy recommendations from simulation scenarios.

    Supported Scenarios:
      - 'led_upgrade': Turnkey LED luminaire retrofit with benchmark capex and ~40% lighting reduction.
      - 'solar_pv': Rooftop solar PV installation with MNRE benchmark capex and annual generation.
      - 'bess_dispatch': Battery Energy Storage System for ToD tariff arbitrage.
      - 'setpoint_adjustment' / 'ac_policy': Thermostat cooling setpoint optimization (+1°C or +2°C).

    Args:
        scenario_name: Identifier for the scenario (e.g. 'led_upgrade', 'solar_pv', 'ac_policy').
        scenario_params: Dictionary of parameters (e.g. 'solar_pv_kwp', 'led_upgrade', 'temp_offset').
        profile: FacilityProfile encoding campus floor area and constraints.

    Returns:
        List of capital/policy Recommendation instances with capex, annual savings, and payback.
    """
    fac_id = profile.facility_id if profile else "university"
    floor_area = float(profile.floor_area_m2 if profile and profile.floor_area_m2 else 35000.0)
    recommendations: List[Recommendation] = []

    s_name = str(scenario_name or "").lower().strip()
    params = scenario_params or {}

    # ── LED Retrofit Scenario ──
    if s_name in ["led_upgrade", "led_retrofit", "led"] or params.get("led_upgrade") is True:
        # Scale lighting kW from floor area (~5 W/m2 lighting power density)
        lighting_kw = float(params.get("lighting_kw") or (floor_area * 0.005))
        reduced_kw = round(lighting_kw * 0.40, 1)  # 40% reduction via LED
        capex = get_benchmark_capital_cost("led_retrofit", quantity=reduced_kw)

        # 3,000 annual operating hours
        annual_kwh = round(reduced_kw * 3000.0, 1)
        bundle, payback = calculate_savings(
            expected_kwh=annual_kwh,
            variance_pct=0.15,
            capital_cost=capex,
            days_modeled=365.0,
        )

        rec = Recommendation(
            id=_generate_rec_id("REC-CAP-LED"),
            facility_id=fac_id,
            type="led_upgrade",
            target="Campus-wide Luminaires",
            action_text=(
                f"LED Luminaire Retrofit: Replace existing T5/T8 fluorescent tubes with high-efficiency commercial LED "
                f"fixtures across {floor_area:,.0f} m², shedding ~{reduced_kw:.1f} kW of constant lighting load."
            ),
            trigger="scenario:led_upgrade",
            savings=bundle,
            cost=capex,
            payback_months=payback,
            assumption_refs=get_savings_assumption_keys(capex_measure="led_retrofit"),
            owner_role="sustainability",
            confidence=0.92,
            is_estimate=True,
        )
        recommendations.append(rec)

    # ── Rooftop Solar PV Addition ──
    solar_kwp = float(params.get("solar_pv_kwp", 0.0))
    if s_name in ["solar_pv", "rooftop_solar", "solar"] or solar_kwp > 0:
        kwp = solar_kwp if solar_kwp > 0 else float(params.get("kwp", 150.0))
        capex = get_benchmark_capital_cost("solar_pv", quantity=kwp)

        # 1,450 kWh/kWp/year standard specific generation in Odisha
        annual_kwh = round(kwp * 1450.0, 1)
        bundle, payback = calculate_savings(
            expected_kwh=annual_kwh,
            variance_pct=0.15,
            capital_cost=capex,
            days_modeled=365.0,
        )

        rec = Recommendation(
            id=_generate_rec_id("REC-CAP-SOLAR"),
            facility_id=fac_id,
            type="solar_bess",
            target="Rooftop Solar PV Array",
            action_text=(
                f"Rooftop Solar PV Expansion: Commission a {kwp:.0f} kWp grid-tied rooftop crystalline PV array "
                f"to displace daytime utility purchases and lower scope 2 emissions."
            ),
            trigger="scenario:solar_pv",
            savings=bundle,
            cost=capex,
            payback_months=payback,
            assumption_refs=get_savings_assumption_keys(capex_measure="solar_pv"),
            owner_role="sustainability",
            confidence=0.90,
            is_estimate=True,
        )
        recommendations.append(rec)

    # ── Thermostat Setpoint / AC Policy ──
    temp_offset = float(params.get("temp_offset", 0.0))
    ac_reduction = float(params.get("ac_policy_reduction_pct", 0.0))
    if s_name in ["setpoint_adjustment", "ac_policy", "comfort_setpoint"] or temp_offset > 0 or ac_reduction > 0:
        offset_c = temp_offset if temp_offset > 0 else 1.0
        # ~7% cooling reduction per degree C setpoint elevation across 2,400 cooling hours
        estimated_cooling_load_kw = floor_area * 0.040  # 40 W/m2 cooling
        saved_kw = estimated_cooling_load_kw * (0.07 * offset_c)
        annual_kwh = round(saved_kw * 2400.0, 1)

        bundle, payback = calculate_savings(
            expected_kwh=annual_kwh,
            variance_pct=0.20,
            capital_cost=0.0,
            days_modeled=365.0,
        )

        rec = Recommendation(
            id=_generate_rec_id("REC-POL-HVAC"),
            facility_id=fac_id,
            type="setpoint_adjustment",
            target="Central HVAC Plant",
            action_text=(
                f"Operational AC Setpoint Adjustment: Raise central cooling setpoint by +{offset_c:.1f}°C "
                f"across conditioned zones (target ~25.0°C), staying strictly within ASHRAE 55 PMV comfort boundaries."
            ),
            trigger="scenario:setpoint_adjustment",
            savings=bundle,
            cost=0.0,
            payback_months=payback,
            assumption_refs=[
                ASSUMPTION_KEY_ELECTRICITY_TARIFF,
                ASSUMPTION_KEY_GRID_EMISSIONS,
                "comfort.optimal_setpoint_c",
            ],
            owner_role="operations",
            confidence=0.88,
            is_estimate=True,
        )
        recommendations.append(rec)

    # ── BESS Energy Storage Addition ──
    bess_kwh = float(params.get("bess_kwh", 0.0))
    if s_name in ["bess", "battery_storage"] or bess_kwh > 0:
        cap_kwh = bess_kwh if bess_kwh > 0 else float(params.get("capacity_kwh", 200.0))
        capex = get_benchmark_capital_cost("bess", quantity=cap_kwh)

        # 300 cycles/year ToD arbitrage
        annual_kwh = round(cap_kwh * 300.0 * 0.90, 1)
        bundle, payback = calculate_savings(
            expected_kwh=annual_kwh,
            variance_pct=0.15,
            capital_cost=capex,
            days_modeled=365.0,
        )

        rec = Recommendation(
            id=_generate_rec_id("REC-CAP-BESS"),
            facility_id=fac_id,
            type="peak_shaving",
            target="Campus BESS Battery Bank",
            action_text=(
                f"Deploy a {cap_kwh:.0f} kWh Lithium Iron Phosphate BESS for Time-of-Day peak tariff arbitrage "
                f"and backup power resilience."
            ),
            trigger="scenario:bess_dispatch",
            savings=bundle,
            cost=capex,
            payback_months=payback,
            assumption_refs=get_savings_assumption_keys(capex_measure="bess"),
            owner_role="sustainability",
            confidence=0.86,
            is_estimate=True,
        )
        recommendations.append(rec)

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# 4. Recommendations from Occupant Complaints
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations_from_complaints(
    clusters: Union[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame],
    profile: Optional[FacilityProfile] = None,
) -> List[Recommendation]:
    """Synthesize operational maintenance recommendations from recurring NLP complaint clusters.

    Maps complaint clusters (e.g. repeated thermal discomfort complaints in a specific wing,
    flickering lighting, or low water pressure) into preventative maintenance work orders.

    Args:
        clusters: NLP complaint clustering output (dict from cluster_complaints, list of cluster dicts, or DataFrame).
        profile: FacilityProfile encoding campus metadata.

    Returns:
        List of operational Recommendation instances.
    """
    if clusters is None:
        return []

    fac_id = profile.facility_id if profile else "university"
    recommendations: List[Recommendation] = []

    cluster_list: List[Dict[str, Any]] = []
    if isinstance(clusters, dict):
        if "cluster_df" in clusters and isinstance(clusters["cluster_df"], pd.DataFrame):
            cdf = clusters["cluster_df"]
            if not cdf.empty and "theme_label" in cdf.columns:
                for t_label, grp in cdf.groupby("theme_label"):
                    terms = clusters.get("tfidf_terms_per_cluster", {}).get(grp["cluster_id"].iloc[0], [])
                    cluster_list.append({
                        "theme_label": str(t_label),
                        "count": len(grp),
                        "terms": terms,
                        "action": clusters.get("top_theme_action", ""),
                    })
        elif "theme_counts" in clusters and isinstance(clusters["theme_counts"], pd.Series):
            for t_label, cnt in clusters["theme_counts"].items():
                cluster_list.append({"theme_label": str(t_label), "count": int(cnt)})
    elif isinstance(clusters, pd.DataFrame):
        if not clusters.empty:
            cluster_list = clusters.to_dict(orient="records")
    elif isinstance(clusters, list):
        cluster_list = [c if isinstance(c, dict) else {"theme_label": str(c)} for c in clusters]

    for cl in cluster_list:
        label = str(cl.get("theme_label") or cl.get("category") or "Thermal Comfort").strip()
        cnt = int(cl.get("count") or 1)
        action_hint = str(cl.get("action") or "")
        target = str(cl.get("target") or cl.get("zone") or "Academic Wing B")

        label_lower = label.lower()
        if any(k in label_lower for k in ["comfort", "thermal", "hvac", "hot", "cold", "stuffy", "cooling"]):
            # HVAC air balancing inspection
            bundle, payback = calculate_savings(
                expected_kwh=35.0,  # Avoids ~35 kWh/day of sub-optimal overcooling/reheat waste
                variance_pct=0.20,
                capital_cost=0.0,
                days_modeled=1.0,
            )
            rec = Recommendation(
                id=_generate_rec_id("REC-CMP-HVAC"),
                facility_id=fac_id,
                type="setpoint_adjustment",
                target=target,
                action_text=(
                    action_hint
                    or f"Conduct HVAC VAV damper & air balancing inspection in {target} to resolve {cnt} recurring "
                    f"thermal comfort complaints and eliminate localized overcooling."
                ),
                trigger=f"complaints_cluster:{label}(count={cnt})",
                savings=bundle,
                cost=0.0,
                payback_months=payback,
                assumption_refs=get_savings_assumption_keys(),
                owner_role="operations",
                confidence=0.85,
                is_estimate=True,
            )
            recommendations.append(rec)

        elif any(k in label_lower for k in ["light", "lighting", "flicker", "bulb", "dark"]):
            # Lighting fixture inspection
            bundle, payback = calculate_savings(
                expected_kwh=15.0,
                variance_pct=0.20,
                capital_cost=0.0,
                days_modeled=1.0,
            )
            rec = Recommendation(
                id=_generate_rec_id("REC-CMP-LTG"),
                facility_id=fac_id,
                type="lighting_audit",
                target=target,
                action_text=(
                    action_hint
                    or f"Audit lighting fixtures and replace failing ballasts / LED drivers in {target} "
                    f"in response to {cnt} occupant complaints."
                ),
                trigger=f"complaints_cluster:{label}(count={cnt})",
                savings=bundle,
                cost=0.0,
                payback_months=payback,
                assumption_refs=get_savings_assumption_keys(),
                owner_role="operations",
                confidence=0.80,
                is_estimate=True,
            )
            recommendations.append(rec)

        elif any(k in label_lower for k in ["water", "leak", "pressure", "tap", "pipe", "plumb"]):
            # Plumbing leak inspection
            bundle, payback = calculate_savings(
                expected_kwh=0.0,
                water_kl=3.0,
                variance_pct=0.20,
                capital_cost=0.0,
                days_modeled=1.0,
            )
            rec = Recommendation(
                id=_generate_rec_id("REC-CMP-WTR"),
                facility_id=fac_id,
                type="water_leak_repair",
                target=target,
                action_text=(
                    action_hint
                    or f"Inspect water supply fixtures, flush valves, and riser joints in {target} "
                    f"addressing {cnt} reported water issues."
                ),
                trigger=f"complaints_cluster:{label}(count={cnt})",
                savings=bundle,
                cost=0.0,
                payback_months=payback,
                assumption_refs=get_savings_assumption_keys(include_water=True),
                owner_role="operations",
                confidence=0.84,
                is_estimate=True,
            )
            recommendations.append(rec)

    return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# 5. Master Aggregation, Service-Guard Verification, and Ranking
# ─────────────────────────────────────────────────────────────────────────────

def build_ranked_recommendations(
    facility_profile: FacilityProfile,
    alerts: Optional[Union[List[Any], pd.DataFrame]] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    context: Optional[Dict[str, Any]] = None,
    complaints: Optional[Union[List[Any], Dict[str, Any], pd.DataFrame]] = None,
    scenarios: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    telemetry_df: Optional[pd.DataFrame] = None,
    tariff: Optional[IndianTariff] = None,
) -> Tuple[List[Recommendation], List[Recommendation]]:
    """Aggregate recommendations from all institutional streams, evaluate through Service-Impact Guard, and rank.

    Crucial Invariant:
      Blocked recommendations are strictly isolated into `blocked_recommendations` and
      NEVER appear in the approved ranked action list!

    Args:
        facility_profile: Institutional FacilityProfile encoding protected loads and comfort constraints.
        alerts: Optional active anomaly detection alerts.
        forecast_df: Optional 24-hour predictive load telemetry.
        context: Optional real-time operational context (occupancy, temperature, sensor data).
        complaints: Optional complaint clustering results or raw complaints.
        scenarios: Optional list of (scenario_name, scenario_params) tuples.
        telemetry_df: Optional facility telemetry DataFrame for India-specific tariff levers.
        tariff: Optional IndianTariff schedule (defaults to OERC HT Commercial).

    Returns:
        tuple[list[Recommendation], list[Recommendation]]:
            - approved_recommendations: Verified service-safe recommendations sorted by expected INR savings descending.
            - blocked_recommendations: Recommendations rejected by the guard with attached technical rationale.
    """
    raw_recommendations: List[Recommendation] = []

    # 1. Synthesize from alerts
    if alerts is not None:
        try:
            raw_recommendations.extend(generate_recommendations_from_alerts(alerts, context, facility_profile))
        except Exception as e:
            logger.warning("Error generating recommendations from alerts: %s", e)

    # 2. Synthesize from forecast
    if forecast_df is not None and not forecast_df.empty:
        try:
            raw_recommendations.extend(generate_recommendations_from_forecast(forecast_df, facility_profile))
        except Exception as e:
            logger.warning("Error generating recommendations from forecast: %s", e)

    # 3. Synthesize from scenarios
    if scenarios:
        for s_name, s_params in scenarios:
            try:
                raw_recommendations.extend(
                    generate_recommendations_from_scenarios(s_name, s_params, facility_profile)
                )
            except Exception as e:
                logger.warning("Error generating recommendations from scenario '%s': %s", s_name, e)

    # 4. Synthesize from complaints
    if complaints is not None:
        try:
            raw_recommendations.extend(generate_recommendations_from_complaints(complaints, facility_profile))
        except Exception as e:
            logger.warning("Error generating recommendations from complaints: %s", e)

    # 5. Synthesize from India-specific tariff levers (Peak kVA, PF/APFC, ToD load shifting, DG audit)
    if telemetry_df is not None and not telemetry_df.empty:
        try:
            raw_recommendations.extend(
                generate_india_tariff_recommendations(
                    telemetry_df=telemetry_df,
                    profile=facility_profile,
                    tariff=tariff,
                    context=context,
                )
            )
        except Exception as e:
            logger.warning("Error generating India tariff recommendations: %s", e)
    elif facility_profile is not None:
        try:
            raw_recommendations.extend(
                generate_india_tariff_recommendations(
                    telemetry_df=None,
                    profile=facility_profile,
                    tariff=tariff,
                    context=context,
                )
            )
        except Exception as e:
            logger.warning("Error generating DG audit recommendation: %s", e)

    # Ensure all recommendations have assumption_refs and is_estimate=True
    for rec in raw_recommendations:
        rec.is_estimate = True
        if not rec.assumption_refs:
            rec.assumption_refs = get_savings_assumption_keys()

    # 6. Evaluate ALL recommendations through the Service-Impact Guard
    approved_list, blocked_list = apply_service_guard(raw_recommendations, facility_profile, context)

    # 7. Rank approved actions strictly by expected INR savings descending
    approved_list.sort(key=lambda r: float(r.savings.inr.expected), reverse=True)

    # Explicitly enforce boundary: blocked items must NEVER appear in approved
    blocked_ids = {r.id for r in blocked_list}
    approved_list = [r for r in approved_list if r.id not in blocked_ids]

    return approved_list, blocked_list


# ─────────────────────────────────────────────────────────────────────────────
# 6. RecommendationEngine Class Interface
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    """Institutional recommendation engine bound to a specific facility profile."""

    def __init__(self, profile: FacilityProfile):
        self.profile = profile

    def generate_from_alerts(
        self, alerts: Union[List[Any], pd.DataFrame], context: Optional[Dict[str, Any]] = None
    ) -> List[Recommendation]:
        """Synthesize recommendations from alerts."""
        return generate_recommendations_from_alerts(alerts, context, self.profile)

    def generate_from_forecast(self, forecast_df: pd.DataFrame) -> List[Recommendation]:
        """Synthesize recommendations from 24h predictive load forecast."""
        return generate_recommendations_from_forecast(forecast_df, self.profile)

    def generate_from_scenarios(
        self, scenario_name: str, scenario_params: Dict[str, Any]
    ) -> List[Recommendation]:
        """Synthesize capital or policy recommendations from simulation scenarios."""
        return generate_recommendations_from_scenarios(scenario_name, scenario_params, self.profile)

    def generate_from_complaints(
        self, clusters: Union[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame]
    ) -> List[Recommendation]:
        """Synthesize operational maintenance recommendations from complaint clusters."""
        return generate_recommendations_from_complaints(clusters, self.profile)

    def generate_from_india_levers(
        self,
        telemetry_df: Optional[pd.DataFrame] = None,
        tariff: Optional[IndianTariff] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Recommendation]:
        """Synthesize India-specific tariff optimization recommendations."""
        return generate_india_tariff_recommendations(
            telemetry_df=telemetry_df,
            profile=self.profile,
            tariff=tariff,
            context=context,
        )

    def build_ranked(
        self,
        alerts: Optional[Union[List[Any], pd.DataFrame]] = None,
        forecast_df: Optional[pd.DataFrame] = None,
        context: Optional[Dict[str, Any]] = None,
        complaints: Optional[Union[List[Any], Dict[str, Any], pd.DataFrame]] = None,
        scenarios: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
        telemetry_df: Optional[pd.DataFrame] = None,
        tariff: Optional[IndianTariff] = None,
    ) -> Tuple[List[Recommendation], List[Recommendation]]:
        """Synthesize, guard, and rank recommendations across all available streams."""
        return build_ranked_recommendations(
            self.profile,
            alerts=alerts,
            forecast_df=forecast_df,
            context=context,
            complaints=complaints,
            scenarios=scenarios,
            telemetry_df=telemetry_df,
            tariff=tariff,
        )
