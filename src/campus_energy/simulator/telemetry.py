"""Telemetry derivation from world state.

Converts true equipment outputs into noisy sensor readings,
applying configured measurement noise while maintaining energy consistency.
"""
import numpy as np
from datetime import datetime

from campus_energy.config import SensorConfig
from campus_energy.simulator.equipment import EquipmentOutput


def apply_sensor_noise(
    true_output: EquipmentOutput,
    sensor_config: SensorConfig,
    rng: np.random.Generator,
) -> dict:
    """Apply realistic sensor noise to a true equipment output.

    Returns a dict with noisy measurements suitable for a telemetry row.
    Maintains: interval_energy_kwh ≈ power_kw / 12 (for 5-min intervals).
    """
    result = {}

    # Power noise: multiplicative (proportional to reading)
    true_power = true_output.power_kw
    if true_power > 0.01:
        noise_factor = 1.0 + rng.normal(0, sensor_config.noise_power_fraction)
        noisy_power = true_power * noise_factor
    else:
        noisy_power = true_power + abs(rng.normal(0, 0.002))
    noisy_power = max(0.0, noisy_power)
    result["power_kw"] = round(noisy_power, 4)

    # Interval energy: derived from noisy power (5-minute interval)
    result["interval_energy_kwh"] = round(noisy_power / 12.0, 6)

    # Voltage noise
    result["voltage_v"] = round(
        sensor_config.nominal_voltage_v + rng.normal(0, sensor_config.noise_voltage_std_v), 2
    )

    # Current: derived from power and voltage for consistency
    voltage = result["voltage_v"]
    pf = sensor_config.nominal_power_factor + rng.normal(0, sensor_config.noise_pf_std)
    pf = float(np.clip(pf, 0.5, 1.0))
    result["power_factor"] = round(pf, 3)

    if voltage > 0 and pf > 0:
        current = (noisy_power * 1000) / (voltage * pf)
        current_noise = 1.0 + rng.normal(0, sensor_config.noise_current_fraction)
        result["current_a"] = round(current * current_noise, 3)
    else:
        result["current_a"] = 0.0

    # State
    result["state"] = true_output.state

    # Flow (pumps only)
    if true_output.flow_lpm is not None:
        if true_output.flow_lpm > 0:
            flow_noise = 1.0 + rng.normal(0, sensor_config.noise_flow_fraction)
            result["flow_lpm"] = round(true_output.flow_lpm * flow_noise, 2)
        else:
            result["flow_lpm"] = 0.0
    else:
        result["flow_lpm"] = None

    # Temperature (HVAC zones)
    if true_output.temperature_c is not None:
        result["temperature_c"] = round(
            true_output.temperature_c + rng.normal(0, sensor_config.noise_temperature_std_c), 2
        )
    else:
        result["temperature_c"] = None

    result["setpoint_c"] = true_output.setpoint_c
    result["sensor_status"] = "healthy"
    result["quality_flag"] = "valid"

    return result
