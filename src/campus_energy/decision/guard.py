"""Service-Impact Guard for Nivara Campus Energy Intelligence Platform.

Enforces the core institutional principle: Energy efficiency recommendations MUST NOT
compromise clinical safety, patient life support, critical cold chains, research integrity,
or occupant thermal comfort.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from campus_energy import assumptions
from campus_energy.analytics.comfort import compute_pmv
from campus_energy.decision.recommendation import GuardResult, Recommendation
from campus_energy.profiles import FacilityProfile, ProtectedLoad


def _find_protected_load(
    profile: FacilityProfile, target: str, context: Optional[Dict[str, Any]] = None
) -> Optional[ProtectedLoad]:
    """Helper to locate any ProtectedLoad matching the recommendation target."""
    if not target:
        return None

    # 1. Exact or case-insensitive match on asset_id
    load = profile.get_protected_load(target)
    if load:
        return load

    # 2. Check context for explicit asset_id
    if context and "asset_id" in context:
        load = profile.get_protected_load(str(context["asset_id"]))
        if load:
            return load

    target_clean = target.strip().lower()

    # 3. Match against protected_loads by zone_id, asset_name, or substring
    for pl in profile.protected_loads:
        if pl.zone_id and pl.zone_id.lower() == target_clean:
            return pl
        if target_clean in pl.asset_id.lower() or pl.asset_id.lower() in target_clean:
            return pl
        if pl.asset_name.lower() in target_clean or target_clean in pl.asset_name.lower():
            return pl

    # 4. Check semantic keywords against load category or reason
    for pl in profile.protected_loads:
        if pl.category == "cold_chain" and any(
            w in target_clean for w in ["cold", "vaccine", "refrig", "blood", "freezer"]
        ):
            return pl
        if pl.category == "life_safety" and any(
            w in target_clean for w in ["icu", "ot", "surgery", "ventilator", "life_support", "dialysis"]
        ):
            return pl
        if pl.category == "critical_it" and any(
            w in target_clean for w in ["server", "datacenter", "ups", "rack"]
        ):
            return pl
        if pl.category == "continuous_process" and any(
            w in target_clean for w in ["process_line", "furnace", "continuous", "line_1"]
        ):
            return pl

    return None


def _is_curtailment_action(rec: Recommendation, context: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if recommendation reduces, sheds, turns off, or reschedules power/runtime."""
    if context and context.get("is_curtailment") is True:
        return True

    # Action types that inherently curtail or shift load
    curtailment_types = {
        "peak_shaving",
        "schedule_shift",
        "equipment_shutdown",
        "curtailment",
        "load_shedding",
        "power_reduction",
        "setpoint_adjustment",
        "demand_response",
    }
    if rec.type.lower() in curtailment_types:
        return True

    action_lower = rec.action_text.lower()
    curtailment_keywords = [
        "curtail",
        "shed",
        "turn off",
        "shut down",
        "shutdown",
        "reduce power",
        "lower power",
        "decrease power",
        "cut power",
        "switch off",
        "power off",
        "reschedule",
        "cycle",
        "duty cycle",
        "duty-cycle",
        "limit power",
        "reduce runtime",
        "standby",
        "turn down",
        "raise setpoint",
        "trip",
        "kill",
    ]
    return any(kw in action_lower for kw in curtailment_keywords)


def _parse_proposed_temperature(
    rec: Recommendation, context: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """Extract or infer proposed temperature in Celsius from context or action text."""
    if context:
        for k in ("proposed_temp_c", "target_temp_c", "setpoint_c", "temp_c", "proposed_setpoint"):
            if k in context and context[k] is not None:
                return float(context[k])
        if "temp_delta_c" in context and context["temp_delta_c"] is not None:
            base = float(context.get("current_temp_c", assumptions.get_value("comfort.optimal_setpoint_c", 24.0)))
            return base + float(context["temp_delta_c"])

    # Regex parse from action text
    text = rec.action_text
    # Matches: "to 28°C", "to 28 C", "to 28.5 degrees", "setpoint to 28°C", "at 28°C"
    m_to = re.search(r"(?:to|at|set\s+to|setpoint\s+(?:to\s+)?)\s*(\d+(?:\.\d+)?)\s*(?:°\s*c|c\b|deg)", text, re.IGNORECASE)
    if m_to:
        return float(m_to.group(1))

    # Matches delta: "+2°C", "+1.5 C", "by 2°C", "raise by 3°C"
    m_delta = re.search(r"(?:\+|by\s+|raise\s+by\s+)(\d+(?:\.\d+)?)\s*(?:°\s*c|c\b|deg)", text, re.IGNORECASE)
    if m_delta:
        base = float(assumptions.get_value("comfort.optimal_setpoint_c", 24.0))
        return base + float(m_delta.group(1))

    # Matches standalone e.g. "28°C"
    m_stand = re.search(r"(\d+(?:\.\d+)?)\s*°\s*c", text, re.IGNORECASE)
    if m_stand:
        return float(m_stand.group(1))

    return None


def check_service_guard(
    recommendation: Recommendation,
    facility_profile: FacilityProfile,
    context: Optional[Dict[str, Any]] = None,
) -> GuardResult:
    """Evaluate an energy recommendation against institutional service-impact constraints.

    Checks performed in sequence:
      1. Protected Loads Check (`protected_load_check`):
         Enforces zero-curtailment on mission-critical assets (ICU, OT, cold-chain, critical IT, continuous process).
      2. Thermal Comfort & ASHRAE PMV Check (`pmv_comfort_check`):
         Validates HVAC setpoints against Predicted Mean Vote (PMV) boundaries for the specific zone.
      3. Scheduled & Detected Occupancy Check (`occupancy_check`):
         Prevents turning off primary lighting or HVAC in occupied zones.
      4. Life-Safety / Critical Ventilation Check (`ventilation_check`):
         Ensures critical air changes, pressurization, and exhaust systems remain uncompromised.

    Args:
        recommendation: Proposed energy recommendation instance.
        facility_profile: Profile of target facility encoding constraints and protected loads.
        context: Optional real-time operational or environmental context (occupancy, temperature, humidity).

    Returns:
        GuardResult with status ("pass" | "blocked" | "needs_review"), reason, constraints checked, and safe alternative.
    """
    constraints_checked: List[str] = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Protected Loads Check
    # ─────────────────────────────────────────────────────────────────────────
    constraints_checked.append("protected_load_check")
    protected_load = _find_protected_load(facility_profile, recommendation.target, context)

    # If target didn't match directly, also inspect action_text for protected assets
    if protected_load is None:
        action_lower = recommendation.action_text.lower()
        for pl in facility_profile.protected_loads:
            if (
                pl.asset_id.lower() in action_lower
                or (pl.zone_id and pl.zone_id.lower() in action_lower)
                or pl.asset_name.lower() in action_lower
            ):
                protected_load = pl
                break

    if protected_load is not None and not protected_load.curtailable:
        if _is_curtailment_action(recommendation, context):
            cat = protected_load.category
            if cat == "cold_chain" or "cold" in protected_load.asset_name.lower() or "vaccine" in protected_load.reason.lower():
                safe_alt = (
                    "Maintain uninterrupted 24/7 power and cooling for vaccine/cold-chain refrigeration. "
                    "Conduct thermal door seal inspection, optimize defrost cycle timing, or utilize "
                    "thermal phase-change buffer storage during peak tariff windows."
                )
            elif cat == "life_safety" or any(w in protected_load.asset_name.lower() for w in ["icu", "ot", "surgery", "ventilator"]):
                safe_alt = (
                    "Maintain continuous uncurtailed operation for life-critical clinical load. "
                    "Perform efficiency audit on upstream central chillers or schedule preventive filter "
                    "maintenance during scheduled facility turnaround."
                )
            elif cat == "critical_it":
                safe_alt = (
                    "Maintain uncurtailed power to server and IT infrastructure. "
                    "Optimize CRAC supply temperature by +0.5°C within ASHRAE TC9.9 guidelines or consolidate idle compute workloads."
                )
            elif cat == "continuous_process":
                safe_alt = (
                    "Maintain uncurtailed operation for continuous process line drives. "
                    "Shift auxiliary raw material staging or schedule process maintenance during planned facility turnaround."
                )
            else:
                safe_alt = (
                    "Maintain uncurtailed operation on target asset; perform efficiency audit on upstream equipment "
                    "or schedule maintenance during scheduled facility turnaround."
                )

            return GuardResult(
                status="blocked",
                reason=(
                    f"Target asset '{recommendation.target}' ({protected_load.asset_name}) is designated as a protected "
                    f"non-curtailable load in {facility_profile.facility_name}. "
                    f"Category: {protected_load.category}. Mandate: {protected_load.reason}."
                ),
                constraints_checked=constraints_checked,
                safe_alternative=safe_alt,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Thermal Comfort & ASHRAE PMV Check
    # ─────────────────────────────────────────────────────────────────────────
    constraints_checked.append("pmv_comfort_check")
    is_hvac_action = (
        recommendation.type in {"setpoint_adjustment", "hvac_optimization", "comfort_setpoint", "temperature_setpoint"}
        or any(k in recommendation.action_text.lower() for k in ["setpoint", "temperature", "thermostat", "raise ac", "increase ac", "cool", "chilled water"])
        or (context and any(k in context for k in ["proposed_temp_c", "target_temp_c", "setpoint_c", "temp_delta_c"]))
    )

    if is_hvac_action:
        zone_id = (context.get("zone_id") or context.get("zone_type") or recommendation.target) if context else recommendation.target
        bounds = facility_profile.get_comfort_bound_for_zone(str(zone_id))

        proposed_temp = _parse_proposed_temperature(recommendation, context)
        if proposed_temp is not None:
            hum = float(context.get("humidity_pct", 50.0)) if context else 50.0
            air_vel = float(context.get("air_velocity_mps", 0.15)) if context else 0.15
            met = float(context.get("metabolic_rate", 1.1)) if context else 1.1
            clo = float(context.get("clothing_clo", 0.6)) if context else 0.6

            predicted_pmv = compute_pmv(
                temp_c=proposed_temp,
                humidity_pct=hum,
                air_velocity_mps=air_vel,
                metabolic_rate=met,
                clothing_clo=clo,
            )

            # Check if zone is unoccupied off-hours
            occ_val = context.get("occupancy") if context else None
            sched_val = context.get("scheduled_occupancy") if context else None
            is_unoccupied = (occ_val == 0 and sched_val == 0) or ("unoccupied" in recommendation.action_text.lower())

            # Comfort boundaries apply during scheduled or detected occupancy
            if not is_unoccupied:
                pmv_violation = predicted_pmv > bounds.max_pmv or predicted_pmv < bounds.min_pmv
                temp_violation = proposed_temp > bounds.temp_max_c or proposed_temp < bounds.temp_min_c

                if pmv_violation or temp_violation:
                    safe_temp = min(bounds.temp_max_c, 25.0)
                    safe_alt = (
                        f"Limit setpoint adjustment to +1.0°C (proposed {safe_temp:.1f}°C keeps predicted PMV "
                        f"within [{bounds.min_pmv:+.2f}, {bounds.max_pmv:+.2f}] envelope) or pre-cool 20 minutes prior to occupancy."
                    )
                    return GuardResult(
                        status="blocked",
                        reason=(
                            f"Proposed HVAC setpoint adjustment to {proposed_temp:.1f}°C results in predicted PMV of {predicted_pmv:+.2f}, "
                            f"exceeding the allowable comfort threshold [{bounds.min_pmv:+.2f}, {bounds.max_pmv:+.2f}] "
                            f"(temperature limits {bounds.temp_min_c:.1f}°C – {bounds.temp_max_c:.1f}°C) for zone '{zone_id}' "
                            f"during scheduled occupancy."
                        ),
                        constraints_checked=constraints_checked,
                        safe_alternative=safe_alt,
                    )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Scheduled & Detected Occupancy Check
    # ─────────────────────────────────────────────────────────────────────────
    constraints_checked.append("occupancy_check")
    action_text_lower = recommendation.action_text.lower()
    is_shutdown_action = (
        recommendation.type in {"equipment_shutdown", "lighting_curtailment", "hvac_shutdown", "lighting_shutdown"}
        or any(k in action_text_lower for k in ["shut off", "turn off", "shutdown", "switch off", "power off", "lights off", "lighting off"])
    )

    if is_shutdown_action:
        # Determine whether zone is currently occupied
        occ_val = context.get("occupancy") if context else None
        sched_val = context.get("scheduled_occupancy") if context else None
        is_occ_flag = context.get("is_occupied") if context else None

        # Check action text for explicit occupancy indicators
        is_explicit_occupied = any(
            w in action_text_lower
            for w in ["in-hours", "during active", "active class", "during lecture", "active hours", "during shift", "occupied classroom"]
        )
        is_explicit_off_hours = any(
            w in action_text_lower
            for w in ["off-hours", "after hours", "unoccupied", "after 22:00", "after 18:00", "night setback", "overnight", "empty"]
        )

        occupied = False
        if is_explicit_occupied:
            occupied = True
        elif is_explicit_off_hours:
            occupied = False
        elif occ_val is not None and occ_val > 0:
            occupied = True
        elif sched_val is not None and sched_val > 0:
            occupied = True
        elif is_occ_flag is True:
            occupied = True

        # If occupied, shutting off primary services (lighting / HVAC) without comfort guarantee is blocked
        is_primary_service = any(k in action_text_lower for k in ["light", "lighting", "ac", "hvac", "cooling", "fan"]) or (
            recommendation.type in {"lighting_curtailment", "hvac_shutdown", "lighting_shutdown"}
        )

        if occupied and is_primary_service:
            occ_desc = f"{occ_val} occupants" if occ_val else ("scheduled class" if is_explicit_occupied else "active human occupancy")
            if "light" in action_text_lower or "lighting" in recommendation.type.lower():
                safe_alt = "Dim lighting to 70% or apply night setback only after scheduled shift ends at 20:00."
            else:
                safe_alt = "Maintain ventilation mode (fan only) or apply moderate setback to 26°C while occupied, scheduling full shutdown after 20:00."

            return GuardResult(
                status="blocked",
                reason=(
                    f"Action proposes shutting off primary services ({'lighting' if 'light' in action_text_lower else 'HVAC'}) "
                    f"in zone '{recommendation.target}' while human occupancy is detected/scheduled ({occ_desc}). "
                    f"Violates minimum service level standard."
                ),
                constraints_checked=constraints_checked,
                safe_alternative=safe_alt,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Life-Safety / Critical Ventilation Check
    # ─────────────────────────────────────────────────────────────────────────
    constraints_checked.append("ventilation_check")
    is_ventilation_action = any(
        k in action_text_lower
        for k in ["ventilation", "exhaust", "air change", "pressurization", "fume hood", "negative pressure", "hepa fan"]
    )
    if is_ventilation_action and _is_curtailment_action(recommendation, context):
        target_lower = recommendation.target.lower()
        is_critical_vent_zone = any(
            w in target_lower or w in action_text_lower
            for w in ["ot", "icu", "operation_theatre", "isolation", "cleanroom", "lab", "toxic", "chemical"]
        )
        if is_critical_vent_zone:
            return GuardResult(
                status="blocked",
                reason=(
                    f"Critical ventilation and pressurization required for infection control / hazardous fume extraction "
                    f"in '{recommendation.target}' cannot be curtailed."
                ),
                constraints_checked=constraints_checked,
                safe_alternative=(
                    "Maintain required air change rates (ACH); inspect HEPA filter differential pressure "
                    "or reduce peripheral corridor lighting instead."
                ),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # All Constraints Passed
    # ─────────────────────────────────────────────────────────────────────────
    return GuardResult(
        status="pass",
        reason="All service constraints satisfied: non-protected asset, PMV thermal comfort within bounds, and occupancy verified.",
        constraints_checked=constraints_checked,
        safe_alternative=None,
    )


def apply_service_guard(
    recommendations: List[Recommendation],
    facility_profile: FacilityProfile,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Recommendation], List[Recommendation]]:
    """Evaluate a batch of recommendations against service constraints.

    Evaluates each recommendation with `check_service_guard`, sets `recommendation.service_guard`,
    and partitions the list into approved and blocked recommendations.

    Args:
        recommendations: List of Recommendation instances to evaluate.
        facility_profile: Institutional FacilityProfile encoding constraints and protected loads.
        context: Optional dictionary containing operational/environmental telemetry.

    Returns:
        tuple[list[Recommendation], list[Recommendation]]: (approved_recommendations, blocked_recommendations)
    """
    approved: List[Recommendation] = []
    blocked: List[Recommendation] = []

    for rec in recommendations:
        # Allow context to have rec-specific override (e.g. context[rec.id] or context[rec.target])
        rec_ctx = context
        if context:
            if rec.id in context and isinstance(context[rec.id], dict):
                rec_ctx = {**context, **context[rec.id]}
            elif rec.target in context and isinstance(context[rec.target], dict):
                rec_ctx = {**context, **context[rec.target]}

        guard_result = check_service_guard(rec, facility_profile, rec_ctx)
        rec.service_guard = guard_result

        if guard_result.status == "blocked":
            blocked.append(rec)
        else:
            approved.append(rec)

    return approved, blocked
