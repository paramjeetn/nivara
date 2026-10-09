"""Configuration loader for simulation parameters."""
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "simulation.yaml"


class TimeSlot(BaseModel):
    start: str
    end: str
    occupancy_fraction: float = Field(ge=0.0, le=1.0)


class Schedule(BaseModel):
    name: str
    working_days: list[int]
    slots: list[TimeSlot]


class EquipmentConfig(BaseModel):
    id: str
    type: str
    name: str
    rated_power_kw: float = Field(ge=0.0)
    equipment_count: int = Field(ge=1)
    unit_power_kw: float = Field(ge=0.0)
    flow_rated_lpm: float | None = None


class ZoneConfig(BaseModel):
    id: str
    name: str
    type: str
    floor: int
    capacity: int = Field(ge=1)
    schedule_id: str
    equipment: list[EquipmentConfig]


class BuildingConfig(BaseModel):
    id: str
    name: str
    type: str
    floors: int
    zones: list[ZoneConfig]
    base_load_kw: float = Field(ge=0.0)


class HolidayConfig(BaseModel):
    date: str
    name: str


class WeatherConfig(BaseModel):
    base_temperature_c: float
    daily_amplitude_c: float
    multi_day_amplitude_c: float
    multi_day_period_days: float
    peak_hour: int
    base_humidity_pct: float
    humidity_amplitude_pct: float
    noise_std_temperature_c: float
    noise_std_humidity_pct: float


class SensorConfig(BaseModel):
    noise_power_fraction: float
    noise_voltage_std_v: float
    noise_current_fraction: float
    noise_pf_std: float
    noise_flow_fraction: float
    noise_temperature_std_c: float
    nominal_voltage_v: float
    nominal_power_factor: float


class SimulationParams(BaseModel):
    start_date: str
    end_date: str
    interval_minutes: int
    timezone: str
    seed: int
    version: str


class ReplayConfig(BaseModel):
    default_speed: int
    batch_size: int


class CampusConfig(BaseModel):
    name: str
    location: str


class SimulationConfig(BaseModel):
    simulation: SimulationParams
    campus: CampusConfig
    buildings: list[BuildingConfig]
    schedules: dict[str, Schedule]
    holidays: list[HolidayConfig]
    weather: WeatherConfig
    sensor: SensorConfig
    replay: ReplayConfig


def load_config(config_path: Path | str | None = None) -> SimulationConfig:
    """Load and validate simulation configuration from YAML."""
    if config_path is None:
        config_path = os.environ.get("SIMULATION_CONFIG", str(DEFAULT_CONFIG_PATH))
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return SimulationConfig(**raw)
