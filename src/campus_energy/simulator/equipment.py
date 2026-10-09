"""Equipment power models for the campus simulator.

Each equipment type has a physically interpretable power model:
- Lighting: rated_power * utilization * state
- HVAC: capacity * duty_cycle(temperature, setpoint, occupancy) * state
- Computer Lab: active_computers * per_unit_load * state
- Pump: rated_power * load_factor * state, with flow output
- Lab Equipment: rated_power * utilization * state

All power values in kW. Flow in liters per minute (lpm).
"""
import numpy as np
from dataclasses import dataclass

from campus_energy.config import EquipmentConfig


@dataclass
class EquipmentOutput:
    """True physical output of an equipment item (before sensor noise)."""
    power_kw: float
    state: str  # ON, OFF
    flow_lpm: float | None = None
    temperature_c: float | None = None
    setpoint_c: float | None = None
    utilization: float = 0.0


def compute_lighting_power(
    equipment: EquipmentConfig,
    is_scheduled: bool,
    occupancy: int,
    rng: np.random.Generator,
) -> EquipmentOutput:
    """Compute lighting power. ON when scheduled AND occupancy > 0."""
    if not is_scheduled or occupancy <= 0:
        standby_kw = 0.005 * equipment.equipment_count
        return EquipmentOutput(power_kw=standby_kw, state="OFF", utilization=0.0)
    utilization = rng.uniform(0.90, 0.98)
    power = equipment.rated_power_kw * utilization
    return EquipmentOutput(power_kw=round(power, 4), state="ON", utilization=utilization)


def compute_hvac_power(
    equipment: EquipmentConfig,
    is_scheduled: bool,
    occupancy: int,
    outdoor_temp_c: float,
    rng: np.random.Generator,
    setpoint_c: float = 24.0,
) -> EquipmentOutput:
    """Compute HVAC/AC power using a duty-cycle model.

    Duty cycle depends on temperature differential and occupancy heat load.
    Bounded [0.3, 1.0] when ON.
    """
    if not is_scheduled or occupancy <= 0:
        standby_kw = 0.01 * equipment.equipment_count
        return EquipmentOutput(
            power_kw=standby_kw, state="OFF",
            temperature_c=outdoor_temp_c, setpoint_c=setpoint_c, utilization=0.0,
        )
    temp_diff = max(0, outdoor_temp_c - setpoint_c)
    temp_duty = 0.3 + 0.7 * min(1.0, temp_diff / 15.0)
    occupancy_heat_kw = occupancy * 0.1
    rated = equipment.rated_power_kw
    occupancy_factor = min(0.15, occupancy_heat_kw / max(rated, 0.1))
    duty_cycle = min(1.0, temp_duty + occupancy_factor)
    duty_cycle = max(0.3, duty_cycle)
    duty_cycle += rng.normal(0, 0.03)
    duty_cycle = float(np.clip(duty_cycle, 0.25, 1.0))
    power = equipment.rated_power_kw * duty_cycle
    indoor_temp = setpoint_c + rng.normal(0, 0.5)
    return EquipmentOutput(
        power_kw=round(power, 4), state="ON",
        temperature_c=round(indoor_temp, 2), setpoint_c=setpoint_c,
        utilization=duty_cycle,
    )


def compute_computer_lab_power(
    equipment: EquipmentConfig,
    is_scheduled: bool,
    occupancy: int,
    zone_capacity: int,
    rng: np.random.Generator,
) -> EquipmentOutput:
    """Compute computer lab power. Active computers proportional to occupancy."""
    total_computers = equipment.equipment_count
    if not is_scheduled or occupancy <= 0:
        standby_fraction = 0.10
        standby_power = total_computers * equipment.unit_power_kw * standby_fraction
        return EquipmentOutput(
            power_kw=round(standby_power, 4), state="OFF", utilization=standby_fraction,
        )
    occupancy_ratio = min(1.0, occupancy / max(zone_capacity, 1))
    active_computers = int(round(total_computers * occupancy_ratio))
    active_power = 0.0
    for _ in range(active_computers):
        load_factor = rng.uniform(0.6, 1.0)
        active_power += equipment.unit_power_kw * load_factor
    standby_computers = total_computers - active_computers
    standby_power = standby_computers * equipment.unit_power_kw * 0.10
    total_power = active_power + standby_power
    utilization = active_computers / max(total_computers, 1)
    return EquipmentOutput(
        power_kw=round(total_power, 4), state="ON", utilization=round(utilization, 4),
    )


def compute_pump_power(
    equipment: EquipmentConfig,
    is_scheduled: bool,
    rng: np.random.Generator,
) -> EquipmentOutput:
    """Compute pump power and flow. ON during scheduled hours."""
    if not is_scheduled:
        return EquipmentOutput(power_kw=0.0, state="OFF", flow_lpm=0.0, utilization=0.0)
    load_factor = rng.uniform(0.75, 0.95)
    power = equipment.rated_power_kw * load_factor
    flow_rated = equipment.flow_rated_lpm or 0.0
    flow_factor = rng.uniform(0.80, 0.98)
    flow = flow_rated * flow_factor
    return EquipmentOutput(
        power_kw=round(power, 4), state="ON",
        flow_lpm=round(flow, 2), utilization=round(load_factor, 4),
    )


def compute_lab_equipment_power(
    equipment: EquipmentConfig,
    is_scheduled: bool,
    occupancy: int,
    zone_capacity: int,
    rng: np.random.Generator,
) -> EquipmentOutput:
    """Compute laboratory equipment power. Utilization based on occupancy."""
    if not is_scheduled or occupancy <= 0:
        standby_power = equipment.rated_power_kw * 0.05
        return EquipmentOutput(
            power_kw=round(standby_power, 4), state="OFF", utilization=0.05,
        )
    occupancy_ratio = min(1.0, occupancy / max(zone_capacity, 1))
    utilization = occupancy_ratio * rng.uniform(0.7, 0.95)
    power = equipment.rated_power_kw * utilization
    return EquipmentOutput(
        power_kw=round(power, 4), state="ON", utilization=round(utilization, 4),
    )


def compute_equipment_power(
    equipment: EquipmentConfig,
    is_scheduled: bool,
    occupancy: int,
    zone_capacity: int,
    outdoor_temp_c: float,
    rng: np.random.Generator,
) -> EquipmentOutput:
    """Route to the appropriate equipment power model based on type."""
    if equipment.type == "lighting":
        return compute_lighting_power(equipment, is_scheduled, occupancy, rng)
    elif equipment.type == "hvac":
        return compute_hvac_power(equipment, is_scheduled, occupancy, outdoor_temp_c, rng)
    elif equipment.type == "computer_lab":
        return compute_computer_lab_power(equipment, is_scheduled, occupancy, zone_capacity, rng)
    elif equipment.type == "pump":
        return compute_pump_power(equipment, is_scheduled, rng)
    elif equipment.type == "lab_equipment":
        return compute_lab_equipment_power(equipment, is_scheduled, occupancy, zone_capacity, rng)
    else:
        raise ValueError(f"Unknown equipment type: {equipment.type}")
