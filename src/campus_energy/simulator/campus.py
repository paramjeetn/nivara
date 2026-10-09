"""Campus orchestrator — generates all world state and telemetry for the simulation.

This is the main simulation loop that:
1. Iterates over every 5-minute timestamp in the 28-day period
2. For each timestamp, generates weather, occupancy, and equipment state
3. Derives telemetry readings with sensor noise
4. Produces context records
5. Tracks cumulative energy for meter readings
6. Builds the asset registry
"""
import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from campus_energy.config import SimulationConfig, BuildingConfig, ZoneConfig, EquipmentConfig
from campus_energy.simulator.schedules import (
    is_working_time,
    get_scheduled_occupancy_fraction,
    get_activity_type,
    is_holiday,
)
from campus_energy.simulator.occupancy import generate_occupancy
from campus_energy.simulator.weather import generate_weather
from campus_energy.simulator.equipment import compute_equipment_power, EquipmentOutput
from campus_energy.simulator.telemetry import apply_sensor_noise


def build_asset_registry(config: SimulationConfig) -> pd.DataFrame:
    """Build the asset registry (assets.csv) from config.

    Creates a hierarchical asset registry:
    - Building assets (monitoring_level=building)
    - Zone assets (monitoring_level=zone)
    - Equipment assets (monitoring_level=equipment)
    - Building meter assets (monitoring_level=building, type=meter)
    """
    assets = []

    for building in config.buildings:
        # Building asset
        assets.append({
            "asset_id": building.id,
            "asset_type": "building",
            "asset_name": building.name,
            "building_id": building.id,
            "floor": None,
            "zone_id": None,
            "parent_asset_id": None,
            "rated_power_kw": None,
            "equipment_count": None,
            "operating_schedule_id": None,
            "monitoring_level": "building",
            "active": True,
        })

        # Building main meter
        meter_id = f"MTR-{building.id}"
        assets.append({
            "asset_id": meter_id,
            "asset_type": "meter",
            "asset_name": f"{building.name} Main Meter",
            "building_id": building.id,
            "floor": None,
            "zone_id": None,
            "parent_asset_id": building.id,
            "rated_power_kw": None,
            "equipment_count": None,
            "operating_schedule_id": None,
            "monitoring_level": "building",
            "active": True,
        })

        for zone in building.zones:
            # Zone asset
            assets.append({
                "asset_id": zone.id,
                "asset_type": "zone",
                "asset_name": zone.name,
                "building_id": building.id,
                "floor": zone.floor,
                "zone_id": zone.id,
                "parent_asset_id": building.id,
                "rated_power_kw": None,
                "equipment_count": None,
                "operating_schedule_id": zone.schedule_id,
                "monitoring_level": "zone",
                "active": True,
            })

            for eq in zone.equipment:
                assets.append({
                    "asset_id": eq.id,
                    "asset_type": eq.type,
                    "asset_name": eq.name,
                    "building_id": building.id,
                    "floor": zone.floor,
                    "zone_id": zone.id,
                    "parent_asset_id": zone.id,
                    "rated_power_kw": eq.rated_power_kw,
                    "equipment_count": eq.equipment_count,
                    "operating_schedule_id": zone.schedule_id,
                    "monitoring_level": "equipment",
                    "active": True,
                })

    return pd.DataFrame(assets)


def generate_timestamps(config: SimulationConfig) -> list[datetime]:
    """Generate all 5-minute timestamps for the simulation period."""
    from dateutil import parser as dtparser
    start = dtparser.parse(config.simulation.start_date).replace(tzinfo=None)
    end = dtparser.parse(config.simulation.end_date).replace(tzinfo=None)
    interval = timedelta(minutes=config.simulation.interval_minutes)
    timestamps = []
    current = start
    while current < end:
        timestamps.append(current)
        current += interval
    return timestamps


def simulate_campus(config: SimulationConfig) -> dict[str, pd.DataFrame]:
    """Run the full normal-world simulation.

    Returns dict with keys: 'assets', 'telemetry', 'context'
    """
    rng = np.random.default_rng(config.simulation.seed)
    timestamps = generate_timestamps(config)
    from dateutil import parser as dtparser
    sim_start = dtparser.parse(config.simulation.start_date).replace(tzinfo=None)

    assets_df = build_asset_registry(config)

    telemetry_rows = []
    context_rows = []

    # Track cumulative energy per equipment asset
    cumulative_energy: dict[str, float] = {}

    reading_counter = 0

    for ts in timestamps:
        # Generate weather for this timestamp
        temp_c, humidity_pct = generate_weather(
            ts, config.weather, sim_start, rng
        )
        holiday = is_holiday(ts, config)
        day_of_week = ts.weekday()  # 0=Monday

        for building in config.buildings:
            building_total_power = 0.0
            building_total_energy = 0.0

            for zone in building.zones:
                # Generate occupancy
                actual_occ, sched_occ = generate_occupancy(
                    ts, zone, zone.schedule_id, config, rng
                )
                is_sched = is_working_time(ts, zone.schedule_id, config)
                activity = get_activity_type(zone.type, zone.schedule_id, ts, config)

                zone_total_power = 0.0

                for eq in zone.equipment:
                    # Compute true equipment power
                    output = compute_equipment_power(
                        eq, is_sched, actual_occ, zone.capacity, temp_c, rng
                    )

                    # Apply sensor noise
                    noisy = apply_sensor_noise(output, config.sensor, rng)

                    # Track cumulative energy
                    if eq.id not in cumulative_energy:
                        cumulative_energy[eq.id] = 0.0
                    cumulative_energy[eq.id] += noisy["interval_energy_kwh"]

                    reading_counter += 1
                    reading_id = f"R-{reading_counter:08d}"

                    telemetry_rows.append({
                        "reading_id": reading_id,
                        "timestamp": ts.isoformat(),
                        "asset_id": eq.id,
                        "reading_type": "equipment",
                        "power_kw": noisy["power_kw"],
                        "interval_energy_kwh": noisy["interval_energy_kwh"],
                        "cumulative_energy_kwh": round(cumulative_energy[eq.id], 6),
                        "voltage_v": noisy["voltage_v"],
                        "current_a": noisy["current_a"],
                        "power_factor": noisy["power_factor"],
                        "state": noisy["state"],
                        "flow_lpm": noisy["flow_lpm"],
                        "temperature_c": noisy["temperature_c"],
                        "setpoint_c": noisy["setpoint_c"],
                        "sensor_status": noisy["sensor_status"],
                        "quality_flag": noisy["quality_flag"],
                    })

                    zone_total_power += noisy["power_kw"]

                # Add base load share per zone (distributed evenly)
                zone_count = len(building.zones)
                zone_base_load = building.base_load_kw / max(zone_count, 1)
                zone_total_power += zone_base_load

                building_total_power += zone_total_power
                building_total_energy += zone_total_power / 12.0

                # Context record for this zone
                context_rows.append({
                    "timestamp": ts.isoformat(),
                    "building_id": building.id,
                    "zone_id": zone.id,
                    "occupancy": actual_occ,
                    "scheduled_occupancy": sched_occ,
                    "occupancy_source": "simulated",
                    "working_hours": is_sched,
                    "holiday": holiday,
                    "academic_activity": activity,
                    "day_of_week": day_of_week,
                    "outdoor_temperature_c": temp_c,
                    "humidity_pct": humidity_pct,
                })

            # Building meter reading (sum of all zone loads including base load)
            meter_id = f"MTR-{building.id}"
            if meter_id not in cumulative_energy:
                cumulative_energy[meter_id] = 0.0
            meter_energy = building_total_energy
            cumulative_energy[meter_id] += meter_energy

            reading_counter += 1
            reading_id = f"R-{reading_counter:08d}"

            # Add small meter-level noise
            meter_noise = 1.0 + rng.normal(0, config.sensor.noise_power_fraction * 0.5)
            metered_power = building_total_power * meter_noise

            telemetry_rows.append({
                "reading_id": reading_id,
                "timestamp": ts.isoformat(),
                "asset_id": meter_id,
                "reading_type": "meter",
                "power_kw": round(metered_power, 4),
                "interval_energy_kwh": round(metered_power / 12.0, 6),
                "cumulative_energy_kwh": round(cumulative_energy[meter_id], 6),
                "voltage_v": round(config.sensor.nominal_voltage_v + rng.normal(0, config.sensor.noise_voltage_std_v), 2),
                "current_a": None,
                "power_factor": None,
                "state": None,
                "flow_lpm": None,
                "temperature_c": None,
                "setpoint_c": None,
                "sensor_status": "healthy",
                "quality_flag": "valid",
            })

    telemetry_df = pd.DataFrame(telemetry_rows)
    context_df = pd.DataFrame(context_rows)

    return {
        "assets": assets_df,
        "telemetry": telemetry_df,
        "context": context_df,
    }
