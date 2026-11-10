"""Realistic Lived-In Operational Seed State for Nivara.

Seeds a populated, believable operational history into the system:
1. Three completed & verified work orders with IPMVP Option B counterfactual math:
   - WO-2025-089 (LED Retrofit in Admin Block): Verified ₹4,800/mo, 565 kWh, 98% hit-rate
   - WO-2025-094 (Pump P1 Cavitation Seal Repair): Verified ₹1,275/mo, 150 kWh, 100% hit-rate
   - WO-2025-098 (Chiller Sensor Recalibration): Verified ₹2,100/mo, 247 kWh, 95% hit-rate
2. Active & Historical Alerts:
   - 2 active alerts (ALT-001 Unoccupied Lighting, ALT-002 Pump Cavitation)
   - 1 snoozed alert (ALT-003 Scheduled Maintenance Check on PC Lab, snoozed until tomorrow)
   - 2 historical resolved alerts (ALT-004 Chiller RTD Spike, ALT-005 Water Header Pressure Drop)
3. Chronological Activity Feed:
   - Relative timestamps: 15 min ago, 4 hours ago, Yesterday, 3 days ago, 1 week ago, 2 weeks ago.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from campus_energy.decision.ledger import SavingsLedger, WorkOrderItem
from campus_energy.decision.recommendation import (
    GuardResult,
    Recommendation,
    SavingsBundle,
    SavingsRange,
)
from campus_energy.schemas import Event, EventType, Severity
from campus_energy.storage.repository import Repository

logger = logging.getLogger(__name__)


def get_seed_work_orders(
    facility_id: str = "university",
    reference_time: Optional[datetime] = None,
) -> List[WorkOrderItem]:
    """Generate the 3 canonical completed & verified work orders for Nivara lived-in baseline."""
    now = reference_time or datetime.now(timezone.utc)

    # 1. WO-2025-089: LED Retrofit in Admin Block
    # Estimated: 576.0 kWh, ₹4,898/mo. Verified: 565.0 kWh, ₹4,800/mo (4800 / 4898 = 98.0% hit rate)
    wo1_created = now - timedelta(days=18)
    wo1_completed = now - timedelta(days=14)
    rec1 = Recommendation(
        id="REC-2025-089",
        facility_id=facility_id,
        type="led_upgrade",
        target="EQ-ADM-LT-01",
        action_text="LED retrofit: replace 72x T8 fluorescent fixtures with high-efficiency 18W LED luminaires in Admin Block",
        trigger="Scheduled Efficiency Audit",
        savings=SavingsBundle(
            kwh=SavingsRange(low=490.0, expected=576.0, high=662.0),
            inr=SavingsRange(low=4163.0, expected=4898.0, high=5632.0),
            co2_kg=SavingsRange(low=401.8, expected=472.3, high=542.8),
        ),
        cost=35000.0,
        payback_months=7.3,
        service_guard=GuardResult(
            status="pass",
            reason="Lighting retrofit improves Lux levels (+120 lux) while maintaining ECBC 2017 Chapter 7 LPD compliance.",
            constraints_checked=["lighting_power_density_check", "occupied_lux_threshold_check"],
        ),
        owner_role="operations",
        lifecycle="verified",
    )
    wo1 = WorkOrderItem(
        work_order_id="WO-2025-089",
        recommendation_id=rec1.id,
        recommendation=rec1,
        status="verified",
        priority="high",
        assignee="Electrical Maintenance Crew",
        created_at=wo1_created,
        sla_deadline=wo1_created + timedelta(hours=48),
        completed_at=wo1_completed,
        estimated_savings=rec1.savings,
        verified_savings=SavingsBundle(
            kwh=SavingsRange(low=536.8, expected=565.0, high=593.2),
            inr=SavingsRange(low=4560.0, expected=4800.0, high=5040.0),
            co2_kg=SavingsRange(low=440.1, expected=463.3, high=486.5),
        ),
        verification_notes="IPMVP Option B Verified: ₹4,800/mo (565.0 kWh). Empirical hit rate: 98%.",
        mv_method="IPMVP Option B / Counterfactual Baseline",
    )

    # 2. WO-2025-094: Pump P1 Cavitation Seal Repair
    # Estimated: 150.0 kWh, ₹1,275/mo. Verified: 150.0 kWh, ₹1,275/mo (1275 / 1275 = 100.0% hit rate)
    wo2_created = now - timedelta(days=10)
    wo2_completed = now - timedelta(days=7)
    rec2 = Recommendation(
        id="REC-2025-094",
        facility_id=facility_id,
        type="equipment_repair",
        target="EQ-UTL-P1-PM",
        action_text="Overhaul suction impeller and replace degraded mechanical shaft seal to eliminate dry-run cavitation on Pump P1",
        trigger="ALT-2025-094",
        savings=SavingsBundle(
            kwh=SavingsRange(low=127.5, expected=150.0, high=172.5),
            inr=SavingsRange(low=1083.8, expected=1275.0, high=1466.2),
            co2_kg=SavingsRange(low=104.6, expected=123.0, high=141.5),
        ),
        cost=8500.0,
        payback_months=6.7,
        service_guard=GuardResult(
            status="pass",
            reason="Service scheduled during pump alternation window with redundant backup P2 online.",
            constraints_checked=["hydraulics_head_pressure_check", "redundancy_reserve_check"],
        ),
        owner_role="operations",
        lifecycle="verified",
    )
    wo2 = WorkOrderItem(
        work_order_id="WO-2025-094",
        recommendation_id=rec2.id,
        recommendation=rec2,
        status="verified",
        priority="critical",
        assignee="Plumbing & Hydraulics Team",
        created_at=wo2_created,
        sla_deadline=wo2_created + timedelta(hours=24),
        completed_at=wo2_completed,
        estimated_savings=rec2.savings,
        verified_savings=SavingsBundle(
            kwh=SavingsRange(low=142.5, expected=150.0, high=157.5),
            inr=SavingsRange(low=1211.2, expected=1275.0, high=1338.8),
            co2_kg=SavingsRange(low=116.8, expected=123.0, high=129.2),
        ),
        verification_notes="IPMVP Option B Verified: ₹1,275/mo (150.0 kWh). Empirical hit rate: 100%.",
        mv_method="IPMVP Option B / Counterfactual Baseline",
    )

    # 3. WO-2025-098: Chiller Sensor Recalibration
    # Estimated: 260.0 kWh, ₹2,210/mo. Verified: 247.0 kWh, ₹2,100/mo (2100 / 2210 = 95.0% hit rate)
    wo3_created = now - timedelta(days=5)
    wo3_completed = now - timedelta(days=3)
    rec3 = Recommendation(
        id="REC-2025-098",
        facility_id=facility_id,
        type="setpoint_adjustment",
        target="CHILLER-01",
        action_text="Recalibrate chilled water return temperature sensor and tune PID valve hunting loop",
        trigger="ALT-2025-098",
        savings=SavingsBundle(
            kwh=SavingsRange(low=221.0, expected=260.0, high=299.0),
            inr=SavingsRange(low=1878.5, expected=2210.0, high=2541.5),
            co2_kg=SavingsRange(low=181.2, expected=213.2, high=245.2),
        ),
        cost=4200.0,
        payback_months=2.0,
        service_guard=GuardResult(
            status="pass",
            reason="Recalibration verified with calibrated Fluke immersion thermometer; supply remains within 7.0°C setpoint.",
            constraints_checked=["chilled_water_delta_t_check", "cop_efficiency_check"],
        ),
        owner_role="operations",
        lifecycle="verified",
    )
    wo3 = WorkOrderItem(
        work_order_id="WO-2025-098",
        recommendation_id=rec3.id,
        recommendation=rec3,
        status="verified",
        priority="medium",
        assignee="HVAC Automation Specialist",
        created_at=wo3_created,
        sla_deadline=wo3_created + timedelta(hours=24),
        completed_at=wo3_completed,
        estimated_savings=rec3.savings,
        verified_savings=SavingsBundle(
            kwh=SavingsRange(low=234.6, expected=247.0, high=259.4),
            inr=SavingsRange(low=1995.0, expected=2100.0, high=2205.0),
            co2_kg=SavingsRange(low=192.4, expected=202.5, high=212.6),
        ),
        verification_notes="IPMVP Option B Verified: ₹2,100/mo (247.0 kWh). Empirical hit rate: 95%.",
        mv_method="IPMVP Option B / Counterfactual Baseline",
    )

    return [wo1, wo2, wo3]


def get_seed_alerts(
    reference_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Return the 5 canonical seed alerts: 2 active, 1 snoozed, 2 resolved."""
    now = reference_time or datetime.now(timezone.utc)

    return [
        {
            "alert_id": "ALT-001",
            "event_id": "ALT-001",
            "asset_id": "EQ-A-CR101-LT",
            "building_id": "BLD-A",
            "zone_id": "CR101",
            "anomaly_type": "lights_left_on",
            "severity": "medium",
            "detected_at": now - timedelta(hours=4),
            "start_time": (now - timedelta(hours=4, minutes=15)).isoformat(),
            "end_time": (now + timedelta(hours=2)).isoformat(),
            "actual_value": 18.5,
            "expected_value": 0.0,
            "deviation": 18.5,
            "confidence": 0.94,
            "status": "active",
            "evidence": "Classroom 101 lighting active after curfew (18.5 kW observed vs 0.0 kW expected with 0 occupants).",
            "recommendation": "Automate after-hours occupancy sensor cutoff on Lighting Panel LP-CR101.",
            "description": "Classroom 101 lighting active after curfew (18.5 kW observed vs 0.0 kW expected).",
        },
        {
            "alert_id": "ALT-002",
            "event_id": "ALT-002",
            "asset_id": "EQ-UTL-P1-PM",
            "building_id": "BLD-UTL",
            "zone_id": "PUMP-RM",
            "anomaly_type": "pump_no_flow",
            "severity": "high",
            "detected_at": now - timedelta(minutes=15),
            "start_time": (now - timedelta(minutes=20)).isoformat(),
            "end_time": (now + timedelta(hours=1)).isoformat(),
            "actual_value": 22.4,
            "expected_value": 0.0,
            "deviation": 22.4,
            "confidence": 0.98,
            "status": "active",
            "evidence": "Utility Water Pump 1 drawing 22.4 kW with 0.0 LPM water flow (Dry run / impeller cavitation detected).",
            "recommendation": "De-energize pump immediately and check suction strainer valve for vapor lock.",
            "description": "Utility Water Pump 1 drawing 22.4 kW with 0.0 LPM flow (Dry run / impeller cavitation detected).",
        },
        {
            "alert_id": "ALT-003",
            "event_id": "ALT-003",
            "asset_id": "EQ-LIB-CA-PC",
            "building_id": "BLD-LIB",
            "zone_id": "LAB-01",
            "anomaly_type": "high_consumption",
            "severity": "low",
            "detected_at": now - timedelta(hours=28),
            "start_time": (now - timedelta(hours=28)).isoformat(),
            "end_time": (now + timedelta(hours=24)).isoformat(),
            "actual_value": 14.2,
            "expected_value": 12.0,
            "deviation": 2.2,
            "confidence": 0.88,
            "status": "snoozed",
            "snooze_until": now + timedelta(days=1),
            "evidence": "Scheduled quarterly power audit flag on Central PC Lab computing workstations.",
            "recommendation": "Verify power management sleep state policies across workstation cluster.",
            "description": "Scheduled quarterly power audit flag on Central PC Lab computing workstations.",
        },
        {
            "alert_id": "ALT-004",
            "event_id": "ALT-004",
            "asset_id": "CHILLER-01",
            "building_id": "BLD-ADM",
            "zone_id": "CENTRAL-PLANT",
            "anomaly_type": "sensor_dropout",
            "severity": "medium",
            "detected_at": now - timedelta(days=5),
            "resolved_at": now - timedelta(days=4),
            "start_time": (now - timedelta(days=5)).isoformat(),
            "end_time": (now - timedelta(days=4)).isoformat(),
            "actual_value": 45.0,
            "expected_value": 8.5,
            "deviation": 36.5,
            "confidence": 0.91,
            "status": "resolved",
            "evidence": "Chiller chilled water return temperature sensor transient spike to 45°C resolved by probe replacement.",
            "recommendation": "Clean thermistor contacts and replace RTD sensor probe.",
            "description": "Chiller chilled water return sensor transient spike resolved by probe replacement.",
        },
        {
            "alert_id": "ALT-005",
            "event_id": "ALT-005",
            "asset_id": "MTR-WATER-01",
            "building_id": "BLD-UTL",
            "zone_id": "HEADER-MAIN",
            "anomaly_type": "pipe_leak_overnight",
            "severity": "high",
            "detected_at": now - timedelta(days=8),
            "resolved_at": now - timedelta(days=7),
            "start_time": (now - timedelta(days=8)).isoformat(),
            "end_time": (now - timedelta(days=7)).isoformat(),
            "actual_value": 42.0,
            "expected_value": 2.0,
            "deviation": 40.0,
            "confidence": 0.96,
            "status": "resolved",
            "evidence": "Distribution header pipe pressure drop and flow surge (+35 LPM) overnight resolved by gasket repair.",
            "recommendation": "Isolate branch valve B-3 and repair joint gasket.",
            "description": "Distribution header pipe pressure drop resolved by gasket repair.",
        },
    ]


def get_seed_alert_statuses(
    reference_time: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return runtime alert_statuses dict for Streamlit session state and view triage."""
    now = reference_time or datetime.now(timezone.utc)

    return {
        "ALT-001": {
            "status": "active",
            "timeline": [
                {
                    "timestamp": now - timedelta(hours=4),
                    "event": "Alert ALT-001 raised by Physical Rule Engine (Unoccupied Lighting).",
                }
            ],
        },
        "ALT-002": {
            "status": "active",
            "timeline": [
                {
                    "timestamp": now - timedelta(minutes=15),
                    "event": "Alert ALT-002 raised by Cavitation Interlock (Pump P1 Cavitation / Dry Run).",
                }
            ],
        },
        "ALT-003": {
            "status": "snoozed",
            "snooze_until": now + timedelta(days=1),
            "timeline": [
                {
                    "timestamp": now - timedelta(hours=28),
                    "event": "Alert ALT-003 raised for quarterly PC Lab energy audit.",
                },
                {
                    "timestamp": now - timedelta(hours=26),
                    "event": f"Alert snoozed until tomorrow by operator ({now + timedelta(days=1):%b %d, %H:%M UTC}).",
                },
            ],
        },
        "ALT-004": {
            "status": "resolved",
            "resolved_at": now - timedelta(days=4),
            "timeline": [
                {
                    "timestamp": now - timedelta(days=5),
                    "event": "Alert ALT-004 raised: RTD sensor spike on Chiller 01.",
                },
                {
                    "timestamp": now - timedelta(days=4),
                    "event": "Sensor probe replaced and calibrated. Alert marked resolved.",
                },
            ],
        },
        "ALT-005": {
            "status": "resolved",
            "resolved_at": now - timedelta(days=7),
            "timeline": [
                {
                    "timestamp": now - timedelta(days=8),
                    "event": "Alert ALT-005 raised: Water header pressure drop and +35 LPM surge.",
                },
                {
                    "timestamp": now - timedelta(days=7),
                    "event": "Gasket replaced and distribution line pressure restored. Alert marked resolved.",
                },
            ],
        },
    }


def get_seed_activity_feed(
    reference_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Return chronological event logs with realistic relative timestamps.

    Timestamps:
        - 15 min ago: Alert Raised · EQ-UTL-P1-PM
        - 4 hours ago: Alert Raised · EQ-A-CR101-LT
        - Yesterday: Alert Snoozed · EQ-LIB-CA-PC
        - 3 days ago: Resolved & Verified · WO-2025-098
        - 1 week ago: Resolved & Verified · WO-2025-094
        - 2 weeks ago: Resolved & Verified · WO-2025-089
    """
    now = reference_time or datetime.now(timezone.utc)

    return [
        {
            "timestamp": now - timedelta(minutes=15),
            "event_type": "alert_raised",
            "badge": "🚨 Alert",
            "badge_color": "#ef4444",
            "title": "Alert Raised · EQ-UTL-P1-PM",
            "detail": "Pump 1 drawing 22.4 kW with 0.0 LPM water flow (Cavitation / Dry Run)",
            "relative_hint": "15 min ago",
        },
        {
            "timestamp": now - timedelta(hours=4),
            "event_type": "alert_raised",
            "badge": "🚨 Alert",
            "badge_color": "#ef4444",
            "title": "Alert Raised · EQ-A-CR101-LT",
            "detail": "Classroom 101 lighting active after curfew (18.5 kW vs 0.0 kW expected)",
            "relative_hint": "4 hours ago",
        },
        {
            "timestamp": now - timedelta(hours=26),
            "event_type": "alert_snoozed",
            "badge": "⏱️ Snoozed",
            "badge_color": "#f59e0b",
            "title": "Alert Snoozed · EQ-LIB-CA-PC",
            "detail": "Scheduled quarterly audit on PC Lab snoozed until tomorrow by operator",
            "relative_hint": "Yesterday",
        },
        {
            "timestamp": now - timedelta(days=3),
            "event_type": "fault_resolved",
            "badge": "✅ Verified",
            "badge_color": "#10b981",
            "title": "Resolved & Verified · WO-2025-098",
            "detail": "₹2,100/mo verified avoided cost (247 kWh, 95% hit-rate)",
            "relative_hint": "3 days ago",
        },
        {
            "timestamp": now - timedelta(days=7),
            "event_type": "fault_resolved",
            "badge": "✅ Verified",
            "badge_color": "#10b981",
            "title": "Resolved & Verified · WO-2025-094",
            "detail": "₹1,275/mo verified avoided cost (150 kWh, 100% hit-rate)",
            "relative_hint": "1 week ago",
        },
        {
            "timestamp": now - timedelta(days=14),
            "event_type": "fault_resolved",
            "badge": "✅ Verified",
            "badge_color": "#10b981",
            "title": "Resolved & Verified · WO-2025-089",
            "detail": "₹4,800/mo verified avoided cost (565 kWh, 98% hit-rate)",
            "relative_hint": "2 weeks ago",
        },
    ]


def seed_database(
    repo: Optional[Repository] = None,
    db_path: str = "data/campus.db",
    reference_time: Optional[datetime] = None,
) -> Repository:
    """Initialize SQLite database tables and insert canonical seed alerts."""
    r = repo or Repository(db_path=db_path)
    now = reference_time or datetime.now(timezone.utc)
    alerts = get_seed_alerts(now)

    with r.get_connection() as conn:
        conn.execute("DELETE FROM alerts WHERE alert_id IN ('ALT-001', 'ALT-002', 'ALT-003', 'ALT-004', 'ALT-005')")

    for a in alerts:
        det_str = a["detected_at"].isoformat() if hasattr(a["detected_at"], "isoformat") else str(a["detected_at"])
        r.insert_alert({
            "alert_id": a["alert_id"],
            "detected_at": det_str,
            "event_start_estimate": a["start_time"],
            "asset_id": a["asset_id"],
            "building_id": a["building_id"],
            "zone_id": a["zone_id"],
            "anomaly_type": a["anomaly_type"],
            "severity": a["severity"],
            "actual_value": a["actual_value"],
            "expected_value": a["expected_value"],
            "deviation": a["deviation"],
            "evidence": a["evidence"],
            "confidence": a["confidence"],
            "recommendation": a["recommendation"],
            "status": a["status"],
        })

    return r


def seed_operational_history(
    ledger: Optional[SavingsLedger] = None,
    facility_id: str = "university",
    db_path: str = "data/campus.db",
    reference_time: Optional[datetime] = None,
) -> SavingsLedger:
    """Populate SavingsLedger and database with realistic lived-in operational history.

    Parameters:
        ledger: Target SavingsLedger instance (if None, a new instance is created).
        facility_id: Facility identifier (default: "university").
        db_path: SQLite database path.
        reference_time: Optional reference datetime for deterministic testing.

    Returns:
        Populated SavingsLedger with 3 verified work orders, attached alerts,
        alert_statuses, and activity feed.
    """
    now = reference_time or datetime.now(timezone.utc)
    target_ledger = ledger if ledger is not None else SavingsLedger()

    # 1. Clear any conflicting items with seed IDs in ledger
    for wo_id in ["WO-2025-089", "WO-2025-094", "WO-2025-098"]:
        target_ledger._items.pop(wo_id, None)

    # 2. Add the 3 verified work orders
    seed_wos = get_seed_work_orders(facility_id=facility_id, reference_time=now)
    for wo in seed_wos:
        target_ledger._items[wo.work_order_id] = wo

    # 3. Attach metadata & seed collections onto ledger
    seed_alerts = get_seed_alerts(reference_time=now)
    seed_statuses = get_seed_alert_statuses(reference_time=now)
    seed_feed = get_seed_activity_feed(reference_time=now)

    setattr(target_ledger, "alerts", seed_alerts)
    setattr(target_ledger, "alert_statuses", seed_statuses)
    setattr(target_ledger, "activity_feed", seed_feed)
    setattr(target_ledger, "verified_work_orders", seed_wos)

    # 4. Populate SQLite repository if db path is accessible
    try:
        seed_database(db_path=db_path, reference_time=now)
    except Exception as exc:
        logger.debug("Database seeding deferred or skipped: %s", exc)

    return target_ledger
