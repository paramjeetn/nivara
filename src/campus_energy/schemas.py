"""Pydantic data models for all platform data contracts.

These schemas define the canonical data contracts for:
- Asset registry
- Telemetry readings  
- Context (environmental/operational)
- Events (injected scenarios)
- Anomaly ground truth
- Complaints
- Alerts (detector output)
"""
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# --- Enums ---

class AssetType(str, Enum):
    BUILDING = "building"
    ZONE = "zone"
    METER = "meter"
    LIGHTING = "lighting"
    HVAC = "hvac"
    COMPUTER_LAB = "computer_lab"
    LAB_EQUIPMENT = "lab_equipment"
    PUMP = "pump"


class MonitoringLevel(str, Enum):
    BUILDING = "building"
    ZONE = "zone"
    CIRCUIT = "circuit"
    EQUIPMENT = "equipment"


class EquipmentState(str, Enum):
    ON = "ON"
    OFF = "OFF"
    UNKNOWN = "UNKNOWN"


class SensorStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    OFFLINE = "offline"


class QualityFlag(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    STALE = "stale"
    OUT_OF_RANGE = "out_of_range"
    DUPLICATED = "duplicated"
    CORRUPTED = "corrupted"


class EventType(str, Enum):
    LIGHTS_LEFT_ON = "lights_left_on"
    HVAC_LEFT_ON = "hvac_left_on"
    PUMP_NO_FLOW = "pump_no_flow"
    POWER_SPIKE = "power_spike"
    GRADUAL_INCREASE = "gradual_increase"
    OVERNIGHT_EXCESS = "overnight_excess"
    OCCUPANCY_MISMATCH = "occupancy_mismatch"
    PEER_ANOMALY = "peer_anomaly"
    SENSOR_DROPOUT = "sensor_dropout"
    COMM_FAILURE = "comm_failure"
    STALE_SENSOR = "stale_sensor"
    CORRUPTED_READING = "corrupted_reading"
    DUPLICATE_READING = "duplicate_reading"
    TIMESTAMP_GAP = "timestamp_gap"
    PIPE_LEAK_OVERNIGHT = "pipe_leak_overnight"
    PF_DROP = "pf_drop"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class ComplaintCategory(str, Enum):
    HVAC = "hvac"
    LIGHTING = "lighting"
    POWER = "power"
    EQUIPMENT = "equipment"
    OTHER = "other"


class ComplaintStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


# --- Asset Registry ---

class Asset(BaseModel):
    """One row in assets.csv."""
    asset_id: str
    asset_type: AssetType
    asset_name: str
    building_id: str
    floor: int | None = None
    zone_id: str | None = None
    parent_asset_id: str | None = None
    rated_power_kw: float | None = None
    equipment_count: int | None = None
    operating_schedule_id: str | None = None
    monitoring_level: MonitoringLevel
    active: bool = True


# --- Telemetry ---

class TelemetryReading(BaseModel):
    """One row in telemetry.csv."""
    reading_id: str
    timestamp: datetime
    asset_id: str
    reading_type: str  # e.g., 'meter', 'equipment', 'sensor'
    quality_flag: QualityFlag = QualityFlag.VALID
    sensor_status: SensorStatus = SensorStatus.HEALTHY
    power_kw: float | None = None
    interval_energy_kwh: float | None = None
    cumulative_energy_kwh: float | None = None
    voltage_v: float | None = None
    current_a: float | None = None
    power_factor: float | None = None
    state: EquipmentState | None = None
    flow_lpm: float | None = None
    temperature_c: float | None = None
    setpoint_c: float | None = None

    @field_validator("power_factor")
    @classmethod
    def validate_power_factor(cls, v: float | None, info) -> float | None:
        flag = info.data.get("quality_flag")
        if flag in (QualityFlag.OUT_OF_RANGE, QualityFlag.CORRUPTED):
            return v
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"power_factor must be between 0 and 1, got {v}")
        return v

    @field_validator("power_kw")
    @classmethod
    def validate_power_non_negative(cls, v: float | None, info) -> float | None:
        flag = info.data.get("quality_flag")
        if flag in (QualityFlag.OUT_OF_RANGE, QualityFlag.CORRUPTED):
            return v
        if v is not None and v < 0:
            raise ValueError(f"power_kw must be non-negative, got {v}")
        return v

    @field_validator("interval_energy_kwh")
    @classmethod
    def validate_energy_non_negative(cls, v: float | None, info) -> float | None:
        flag = info.data.get("quality_flag")
        if flag in (QualityFlag.OUT_OF_RANGE, QualityFlag.CORRUPTED):
            return v
        if v is not None and v < 0:
            raise ValueError(f"interval_energy_kwh must be non-negative, got {v}")
        return v


# --- Context ---

class ContextRecord(BaseModel):
    """One row in context.csv."""
    timestamp: datetime
    building_id: str
    zone_id: str | None = None
    occupancy: int = Field(ge=0)
    scheduled_occupancy: int = Field(ge=0)
    occupancy_source: str = "simulated"  # simulated, sensor, timetable
    working_hours: bool = False
    holiday: bool = False
    academic_activity: str | None = None  # lecture, lab, exam, none
    day_of_week: int = Field(ge=0, le=6)  # 0=Mon, 6=Sun
    outdoor_temperature_c: float
    humidity_pct: float = Field(ge=0.0, le=100.0)


# --- Events ---

class Event(BaseModel):
    """One row in events.csv — injected scenarios."""
    event_id: str
    event_type: EventType
    asset_id: str
    start_time: datetime
    end_time: datetime
    severity: Severity
    description: str
    injection_parameters: dict[str, Any] = Field(default_factory=dict)


# --- Ground Truth ---

class AnomalyGroundTruth(BaseModel):
    """One row in anomaly_ground_truth.csv."""
    event_id: str
    asset_id: str
    event_type: EventType
    start_time: datetime
    end_time: datetime
    severity: Severity
    expected_detection: bool = True
    expected_detection_method: str  # rule, statistical, ml
    root_cause: str


# --- Complaints ---

class Complaint(BaseModel):
    """One row in complaints.csv."""
    complaint_id: str
    timestamp: datetime
    building_id: str
    zone_id: str | None = None
    category: ComplaintCategory
    description: str
    status: ComplaintStatus = ComplaintStatus.OPEN
    resolution_timestamp: datetime | None = None


# --- Alerts (detector output) ---

class Alert(BaseModel):
    """Structured alert from the anomaly detector."""
    alert_id: str
    detected_at: datetime
    event_start_estimate: datetime | None = None
    asset_id: str
    building_id: str
    zone_id: str | None = None
    anomaly_type: str
    severity: Severity
    actual_value: float | None = None
    expected_value: float | None = None
    deviation: float | None = None
    evidence: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    recommendation: str
    status: AlertStatus = AlertStatus.ACTIVE


# --- Data Quality Report ---

class DataQualityReport(BaseModel):
    """Data quality evaluation summary for telemetry datasets."""
    total_records: int
    quality_score: float = Field(ge=0.0, le=100.0)
    missing_intervals_count: int = 0
    duplicate_count: int = 0
    stuck_sensor_count: int = 0
    outlier_count: int = 0
    asset_coverage_pct: float = Field(ge=0.0, le=100.0, default=100.0)
    confidence_penalty: float = Field(default=0.0)
    recommendations: list[str] = Field(default_factory=list)
