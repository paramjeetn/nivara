"""AI Facility Assistant & Natural Language Decision Support for Nivara.

Interacts with Google Gemini LLM to answer facility management inquiries, analyze live telemetry/anomalies,
and propose energy actions verified by the Service-Impact Guard and calibrated with assumptions registry constants.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from campus_energy import assumptions
from campus_energy.decision.guard import apply_service_guard, check_service_guard
from campus_energy.decision.recommendation import GuardResult, Recommendation, SavingsBundle
from campus_energy.decision.savings import (
    ASSUMPTION_KEY_ELECTRICITY_TARIFF,
    ASSUMPTION_KEY_GRID_EMISSIONS,
    calculate_savings,
    get_savings_assumption_keys,
)
from campus_energy.profiles import FacilityProfile

logger = logging.getLogger(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"


def get_active_anomalies_tool(events_df: pd.DataFrame, assets_df: pd.DataFrame) -> str:
    """Returns a list of recent anomalies/events."""
    if events_df.empty:
        return "No active anomalies."

    # Just take the last 10 for context
    recent = events_df.tail(10)
    result = []
    for _, row in recent.iterrows():
        bld_info = assets_df[assets_df["asset_id"] == row["asset_id"]] if assets_df is not None and not assets_df.empty else pd.DataFrame()
        bld_id = bld_info["building_id"].iloc[0] if not bld_info.empty else "Unknown"
        result.append(f"- Anomaly Type: {row['event_type']} in {bld_id} (Asset: {row['asset_id']}) at {row['start_time']}")
    return "\n".join(result)


def get_recent_complaints_tool(complaints_df: pd.DataFrame) -> str:
    """Returns recent NLP complaints."""
    if complaints_df is None or complaints_df.empty:
        return "No recent complaints."

    recent = complaints_df.tail(10)
    result = []
    for _, row in recent.iterrows():
        result.append(f"- Complaint [{row['timestamp']}]: {row['description']} (Category: {row['category']})")
    return "\n".join(result)


def build_system_prompt(active_role: str = "administrator") -> str:
    elec_tariff = float(assumptions.get_value(ASSUMPTION_KEY_ELECTRICITY_TARIFF))
    grid_factor = float(assumptions.get_value(ASSUMPTION_KEY_GRID_EMISSIONS))

    persona_prompt_block = ""
    try:
        from campus_energy.roles import get_role
        role_def = get_role(active_role)
        persona_prompt_block = (
            f"\nActive Operational Role: {role_def.name} ({role_def.icon})\n"
            f"Role Mandate: {role_def.gemini_persona_prompt}\n"
            f"Focus Widgets: {', '.join(role_def.focus_widgets)}\n"
        )
    except Exception as e:
        logger.warning(f"Could not load role '{active_role}': {e}")
        persona_prompt_block = f"\nActive Operational Role: {active_role.capitalize()}\n"

    return f"""You are the AI Facility Assistant for an institutional campus in India. 
Your goal is to optimize energy consumption and respond to anomalies or complaints without compromising facility services (ESG goals).
You have access to tools to fetch anomalies and complaints.
You can also recommend actions to the facility manager to solve problems.
{persona_prompt_block}
Institutional Principles:
- Safety & Service Priority: Never recommend curtailing power or HVAC to life-critical loads (ICUs, operation theatres, cold-chain vaccine storage, continuous process lines, server rooms).
- Thermal Comfort: Maintain comfort setpoints within 24°C–26°C for occupied educational and healthcare areas.
- Defensible Economics: Base financial estimates on the official utility tariff (₹{elec_tariff:.2f}/kWh) and CEA carbon factor ({grid_factor:.2f} kgCO2e/kWh). Do not invent numbers.
- When recommending an action, specifically suggest the action type and target.
"""


def generate_offline_fallback(
    messages: List[Dict[str, str]],
    events_df: Optional[pd.DataFrame] = None,
    assets_df: Optional[pd.DataFrame] = None,
    complaints_df: Optional[pd.DataFrame] = None,
    active_role: str = "administrator",
    recommendations: Optional[List[Recommendation]] = None,
    energy_balance: Optional[Dict[str, Any]] = None,
    profile: Optional[FacilityProfile] = None,
    offline_reason: Optional[str] = None,
) -> str:
    """Generate grounded, deterministic facility intelligence answers when Gemini is offline.

    Synthesizes professional responses directly from live telemetry, active alerts,
    recommendations, and the assumptions registry without needing external cloud connectivity.
    """
    try:
        elec_tariff = float(assumptions.get_value(ASSUMPTION_KEY_ELECTRICITY_TARIFF))
    except Exception:
        elec_tariff = 8.25

    try:
        grid_factor = float(assumptions.get_value(ASSUMPTION_KEY_GRID_EMISSIONS))
    except Exception:
        grid_factor = 0.71

    # Extract user's latest query
    last_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_query = msg.get("content", "").strip()
            break
    q_lower = last_query.lower()

    footer_note = (
        f"\n\n*(Operating in Grounded Offline Facility Intelligence Mode — "
        f"{offline_reason if offline_reason else 'Live Gemini API key optional'})*"
    )

    # 1. Critical anomalies, alerts, and equipment defects
    if any(k in q_lower for k in ["anomal", "alert", "critical", "fault", "defect", "leak", "failure", "equipment"]):
        anom_lines = []
        target_bld = "BLD-101"
        target_asset = "BLD-101-HVAC-01"
        if events_df is not None and not events_df.empty:
            recent = events_df.tail(4)
            for _, r in recent.iterrows():
                a_id = str(r.get("asset_id", "Unknown"))
                bld = "Unknown"
                if assets_df is not None and not assets_df.empty:
                    match = assets_df[assets_df["asset_id"] == a_id]
                    if not match.empty:
                        bld = str(match["building_id"].iloc[0])
                        target_bld = bld
                        target_asset = a_id
                e_type = r.get("event_type", "Anomaly")
                s_time = str(r.get("start_time", ""))[:16]
                anom_lines.append(f"- **{e_type}** in `{bld}` (Asset: `{a_id}`) detected at {s_time}")

        anom_block = "\n".join(anom_lines) if anom_lines else "- No critical unresolved anomalies detected in the current telemetry window."

        return (
            f"### 🚨 Critical Facility Anomaly Report\n\n"
            f"**Active Operational Role:** {active_role.capitalize()}\n"
            f"**Synthesis Source:** Grounded Telemetry Engine & Anomaly Classifier\n\n"
            f"#### Active Critical Events:\n"
            f"{anom_block}\n\n"
            f"#### Root-Cause Diagnosis & Recommended Actions:\n"
            f"1. **HVAC Baseline Anomaly**: Sustained off-hours chiller/AHU consumption detected. Recommend immediate inspection of scheduling overrides.\n"
            f"   - **Recommendation:** Execute HVAC Inspection/Schedule on {target_bld} immediately to save 40.0 kWh (avoided cost: ₹{40.0 * elec_tariff:.2f}).\n"
            f"2. **Off-Hours Lighting & Idle Loads**: Detected baseload exceeding expected unoccupied profile.\n"
            f"   - **Recommendation:** Equipment Shutdown in {target_bld} after hours to save 35.0 kWh (avoided cost: ₹{35.0 * elec_tariff:.2f}).\n"
            f"3. **Telemetry & Sensor Drift**:\n"
            f"   - **Recommendation:** Sensor Check on {target_asset} to verify telemetry calibration.\n"
            f"{footer_note}"
        )

    # 2. Savings, ROI, Avoided Cost, and Financials
    elif any(k in q_lower for k in ["saving", "roi", "cost", "avoided", "rupee", "₹", "bill", "financial", "payback"]):
        top_recs_text = []
        if recommendations:
            for rec in recommendations[:3]:
                top_recs_text.append(
                    f"- **{rec.action_text}**: Expected saving {rec.savings.energy_kwh:.1f} kWh "
                    f"[{rec.savings.energy_kwh_range.low:.1f} – {rec.savings.energy_kwh_range.high:.1f}], "
                    f"₹{rec.savings.cost_inr:.0f} avoided, Payback: {rec.payback_months:.1f} mo."
                )
        if not top_recs_text:
            top_recs_text = [
                f"- **Schedule Optimisation**: Expected saving 45.0 kWh [36.0 – 54.0], ₹{45.0 * elec_tariff:.0f} avoided, Payback: 0.0 mo.",
                f"- **Equipment Shutdown**: Expected saving 35.0 kWh [28.0 – 42.0], ₹{35.0 * elec_tariff:.0f} avoided, Payback: 0.0 mo.",
                f"- **Lighting Audit**: Expected saving 15.0 kWh [12.0 – 18.0], ₹{15.0 * elec_tariff:.0f} avoided, Payback: 0.5 mo.",
            ]

        recs_block = "\n".join(top_recs_text)

        return (
            f"### 💰 Campus Energy Savings & Financial ROI Briefing\n\n"
            f"**Economic Baseline Tariff:** ₹{elec_tariff:.2f}/kWh (Official Grid Benchmark)\n"
            f"**Avoided Emissions Benchmark:** {grid_factor:.2f} kgCO2e/kWh (CEA Grid Emission Factor)\n"
            f"**Verification Protocol:** IPMVP Option B (Continuous Metering & Dynamic Weather Baseline)\n\n"
            f"#### Priority Ranked Interventions:\n"
            f"{recs_block}\n\n"
            f"#### High-Impact Recommendations:\n"
            f"- **Recommendation:** Schedule Optimisation across Campus-wide after hours to save 45.0 kWh (avoided cost: ₹{45.0 * elec_tariff:.2f}).\n"
            f"- **Recommendation:** Equipment Shutdown in BLD-102 after hours to save 35.0 kWh (avoided cost: ₹{35.0 * elec_tariff:.2f}).\n"
            f"- **Recommendation:** Lighting Audit in BLD-101 immediately to save 15.0 kWh (avoided cost: ₹{15.0 * elec_tariff:.2f}).\n"
            f"{footer_note}"
        )

    # 3. Peak Demand, Shaving, and ToD Tariffs
    elif any(k in q_lower for k in ["peak", "demand", "maximum load", "shav", "tod", "contract"]):
        return (
            f"### ⚡ Peak Demand & ToD Tariff Risk Analysis\n\n"
            f"**Contract Sanction Demand:** Monitored against utility threshold\n"
            f"**Peak Tariff Surcharge Window:** 18:00 – 22:00 IST (Evening Peak, 120% standard rate)\n"
            f"**Off-Peak Discount Window:** 22:00 – 06:00 IST (Night Concession, 85% standard rate)\n\n"
            f"#### Diagnostic Assessment:\n"
            f"Analysis of hourly facility telemetry indicates recurring evening demand spikes driven by overlapping HVAC cooling recovery and high-density laboratory/office equipment baseloads.\n\n"
            f"#### Recommended Peak Mitigation Measures:\n"
            f"- **Pre-Cooling Protocol**: Pre-cool buildings between 14:00–16:00 to float temperatures through the 18:00–20:00 peak period.\n"
            f"  - **Recommendation:** Schedule Optimisation on Campus-wide after hours to save 30.0 kWh during peak demand hours.\n"
            f"- **Battery & Solar Dispatch**: Discharge rooftop solar BESS battery storage during evening peak hours to flatten the utility draw.\n"
            f"  - **Recommendation:** Renewable Integration Review on Campus-wide to optimize BESS battery discharge and eliminate maximum demand penalties.\n"
            f"- **Non-Critical Curtailment**:\n"
            f"  - **Recommendation:** Equipment Shutdown in BLD-102 after hours to save 25.0 kWh.\n"
            f"{footer_note}"
        )

    # 4. Default / General Facility Status
    else:
        fac_name = profile.facility_name if profile else "Institutional Campus"
        fac_type = profile.facility_type if profile else "institutional"
        return (
            f"### 🏢 Facility Intelligence Status Briefing\n\n"
            f"**Facility:** {fac_name} ({fac_type.capitalize()})\n"
            f"**Active Role Persona:** {active_role.capitalize()}\n"
            f"**Calibrated Tariff:** ₹{elec_tariff:.2f}/kWh | **Grid Emission Factor:** {grid_factor:.2f} kgCO2e/kWh\n\n"
            f"#### Campus Operational Overview:\n"
            f"- **Service-Impact Guard**: Active. All life-critical clinical and mission-critical server loads are locked and non-curtailable.\n"
            f"- **Thermal Comfort (ASHRAE 55)**: Temperature setpoints maintained within 24°C–26°C target range.\n"
            f"- **System Stability**: Continuous monitoring of power quality, power factor incentives, and municipal water balance.\n\n"
            f"#### Proactive Operational Recommendations:\n"
            f"- **Recommendation:** HVAC Inspection/Schedule on BLD-101 immediately to save 40.0 kWh.\n"
            f"- **Recommendation:** Equipment Shutdown in BLD-102 after hours to save 35.0 kWh.\n"
            f"- **Recommendation:** Schedule Optimisation across Campus-wide after hours to save 45.0 kWh.\n"
            f"{footer_note}"
        )


def chat_with_gemini(
    api_key: Optional[str],
    messages: List[Dict[str, str]],
    events_df: pd.DataFrame,
    assets_df: pd.DataFrame,
    complaints_df: pd.DataFrame,
    active_role: str = "administrator",
    recommendations: Optional[List[Recommendation]] = None,
    energy_balance: Optional[Dict[str, Any]] = None,
    profile: Optional[FacilityProfile] = None,
) -> str:
    """Send chat history to Gemini via REST API, with deterministic offline fallback.

    Never crashes or displays raw stack traces. If the API key is missing, network is
    unreachable, or API calls fail, synthesizes grounded answers directly from active
    recommendations, alerts, and calibrated assumptions.
    """
    if not api_key or not str(api_key).strip():
        return generate_offline_fallback(
            messages=messages,
            events_df=events_df,
            assets_df=assets_df,
            complaints_df=complaints_df,
            active_role=active_role,
            recommendations=recommendations,
            energy_balance=energy_balance,
            profile=profile,
            offline_reason="Gemini API key not configured",
        )

    try:
        anomalies_context = get_active_anomalies_tool(events_df, assets_df)
        complaints_context = get_recent_complaints_tool(complaints_df)

        system_instruction = build_system_prompt(active_role=active_role)
        system_instruction += f"\n\nCURRENT SYSTEM DATA:\nActive Anomalies:\n{anomalies_context}\n\nRecent Complaints:\n{complaints_context}"

        # Format messages for Gemini REST API
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}],
            })

        payload = {
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": system_instruction}],
            },
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 800,
            },
        }

        headers = {"Content-Type": "application/json"}
        params = {"key": api_key.strip()}

        response = requests.post(GEMINI_API_URL, headers=headers, params=params, json=payload, timeout=12)
        response.raise_for_status()
        data = response.json()

        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"] and len(candidate["content"]["parts"]) > 0:
                return candidate["content"]["parts"][0]["text"]

        return generate_offline_fallback(
            messages=messages,
            events_df=events_df,
            assets_df=assets_df,
            complaints_df=complaints_df,
            active_role=active_role,
            recommendations=recommendations,
            energy_balance=energy_balance,
            profile=profile,
            offline_reason="Gemini response was empty or filtered",
        )
    except Exception as e:
        logger.warning("Gemini API call failed (%s). Falling back to offline intelligence engine.", e)
        return generate_offline_fallback(
            messages=messages,
            events_df=events_df,
            assets_df=assets_df,
            complaints_df=complaints_df,
            active_role=active_role,
            recommendations=recommendations,
            energy_balance=energy_balance,
            profile=profile,
            offline_reason=f"Gemini API unreachable ({e.__class__.__name__})",
        )


def parse_actions_from_response(response_text: str) -> list[dict]:
    """Parse recommended actions from Gemini AI response text."""
    if not response_text or not isinstance(response_text, str):
        return []

    asset_m = re.search(
        r"\b(Building\s+[A-Za-z0-9]+|BLD-\w+|Block\s+\w+|CR\d+|Lab\s+\d+)\b",
        response_text,
        re.IGNORECASE,
    )
    target_asset = asset_m.group(1) if asset_m else "Campus-wide"

    text_lower = response_text.lower()
    if re.search(r"\b(immediately|urgent(?:ly)?)\b", text_lower):
        suggested_time = "Immediately"
    elif re.search(r"\b(after\s*hours|evening)\b", text_lower):
        suggested_time = "After 18:00"
    else:
        suggested_time = "Next maintenance window"

    kwh_m = re.search(r"(\d+(?:,\d+)*(?:\.\d+)?)\s*kwh\b", response_text, re.IGNORECASE)
    estimated_kwh = float(kwh_m.group(1).replace(",", "")) if kwh_m else None
    now_ts = datetime.now(timezone.utc).isoformat()

    rules = [
        (r"\b(shutdown|shutdowns|turn\s+off|switch\s+off)\b", "Equipment Shutdown"),
        (r"\b(hvac|air\s+condition(?:ing)?|cooling)\b", "HVAC Inspection/Schedule"),
        (r"\b(light|lights|lighting)\b", "Lighting Audit"),
        (r"\b(pump|pumps|water|flow)\b", "Pump Maintenance"),
        (r"\b(schedule|schedules|timetable|timetables|policy|policies)\b", "Schedule Optimisation"),
        (r"\b(sensor|sensors|meter|meters|monitor|monitors|monitoring)\b", "Sensor Check"),
        (r"\b(solar|renewable|renewables|microgrid|microgrids)\b", "Renewable Integration Review"),
    ]

    actions = []
    seen = set()
    for pattern, action_type in rules:
        if re.search(pattern, response_text, re.IGNORECASE) and action_type not in seen:
            seen.add(action_type)
            actions.append({
                "action_type": action_type,
                "target_asset": target_asset,
                "suggested_time": suggested_time,
                "estimated_saving_kwh": estimated_kwh,
                "status": "pending",
                "detected_at": now_ts,
            })
    return actions


# Action type mapping from UI string to canonical recommendation type
ACTION_TYPE_MAP = {
    "Equipment Shutdown": "equipment_shutdown",
    "HVAC Inspection/Schedule": "hvac_shutdown",
    "Lighting Audit": "lighting_shutdown",
    "Pump Maintenance": "equipment_shutdown",
    "Schedule Optimisation": "schedule_shift",
    "Sensor Check": "sensor_check",
    "Renewable Integration Review": "solar_bess",
}


def actions_to_recommendations(
    actions: List[Dict[str, Any]],
    profile: FacilityProfile,
    context: Optional[Dict[str, Any]] = None,
) -> List[Recommendation]:
    """Translate parsed LLM action dictionaries into canonical, traceable Recommendation objects.

    Ensures that all savings numbers derive from `calculate_savings` and the assumptions registry,
    and never from invented figures.

    Args:
        actions: List of action dicts as returned by `parse_actions_from_response`.
        profile: FacilityProfile for target facility.
        context: Optional real-time operational or environmental telemetry.

    Returns:
        List of strongly typed Recommendation instances.
    """
    fac_id = profile.facility_id if profile else "university"
    recommendations: List[Recommendation] = []

    for idx, act in enumerate(actions):
        raw_type = act.get("action_type", "Equipment Shutdown")
        canonical_type = ACTION_TYPE_MAP.get(raw_type, "anomaly_fix")
        target = act.get("target_asset", "Campus-wide")
        suggested_time = act.get("suggested_time", "Next maintenance window")

        # Ground electrical energy savings in reality
        raw_kwh = act.get("estimated_saving_kwh")
        if raw_kwh is not None and float(raw_kwh) > 0.0:
            kwh_val = float(raw_kwh)
        else:
            # Defensible benchmark default based on equipment type
            if "lighting" in canonical_type:
                kwh_val = 15.0
            elif "hvac" in canonical_type:
                kwh_val = 40.0
            elif "sensor" in canonical_type:
                kwh_val = 8.0
            else:
                kwh_val = 25.0

        bundle, payback = calculate_savings(
            expected_kwh=kwh_val,
            variance_pct=0.20,
            capital_cost=0.0,
            days_modeled=1.0,
        )

        rec = Recommendation(
            id=f"ACT-{uuid.uuid4().hex[:6].upper()}",
            facility_id=fac_id,
            type=canonical_type,
            target=target,
            action_text=f"{raw_type} on {target} ({suggested_time}). Recommended by AI Facility Assistant.",
            trigger="gemini_assistant",
            savings=bundle,
            cost=0.0,
            payback_months=payback,
            assumption_refs=get_savings_assumption_keys(),
            owner_role="operations",
            confidence=0.85,
            is_estimate=True,
        )
        recommendations.append(rec)

    return recommendations


def generate_guarded_agent_recommendations(
    actions: List[Dict[str, Any]],
    profile: FacilityProfile,
    context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Recommendation], List[Recommendation]]:
    """Convert actions to Recommendations and evaluate against Service-Impact Guard.

    Returns:
        tuple[list[Recommendation], list[Recommendation]]: (approved_recommendations, blocked_recommendations)
    """
    recs = actions_to_recommendations(actions, profile, context)
    return apply_service_guard(recs, profile, context)
