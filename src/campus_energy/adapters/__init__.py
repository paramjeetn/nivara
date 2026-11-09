"""Adapters package for Nivara Campus Energy platform.

Provides adapters for:
- External telemetry ingestion & smart mapping (Bring-Your-Own-Data / CSV Upload)
- Data Quality Engine and anomaly confidence calibration
"""
from campus_energy.adapters.quality import (
    DataQualityEngine,
    evaluate_telemetry_quality,
)
from campus_energy.adapters.csv_upload import (
    DataIngestionError,
    detect_column_mapping,
    detect_unit_conversions,
    normalize_timezones,
    process_uploaded_csv,
)
from campus_energy.schemas import DataQualityReport

__all__ = [
    "DataQualityEngine",
    "DataQualityReport",
    "evaluate_telemetry_quality",
    "DataIngestionError",
    "detect_column_mapping",
    "detect_unit_conversions",
    "normalize_timezones",
    "process_uploaded_csv",
]
