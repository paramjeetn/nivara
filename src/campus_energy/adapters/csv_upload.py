"""Bring-Your-Own-Data (BYOD) CSV Ingestion and Smart Mapping Adapter.

Ingests external/third-party CSV telemetry files with varying headers and formats:
- Smart heuristic column mapping (regex / substring / alias matching)
- Custom explicit column mapping dictionary override
- Unit conversions (e.g., Watts -> kW, Joules -> kWh, Wh -> kWh)
- Timezone normalization (UTC, Asia/Kolkata / IST, or naive timestamps)
- Schema validation against TelemetryReading Pydantic model
- Data quality evaluation producing DataQualityReport
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
import re
from typing import Any
import uuid
import warnings

import numpy as np
import pandas as pd
from pydantic import ValidationError

from campus_energy.adapters.quality import DataQualityEngine
from campus_energy.schemas import (
    DataQualityReport,
    QualityFlag,
    SensorStatus,
    TelemetryReading,
)

logger = logging.getLogger(__name__)


class DataIngestionError(ValueError):
    """Raised when an uploaded telemetry CSV is malformed or invalid."""
    pass


# Canonical heuristic patterns
COLUMN_PATTERNS: dict[str, list[str]] = {
    "timestamp": [
        r"^time(stamp)?$",
        r"^date(_?time)?$",
        r"^recorded_?at$",
        r"^record_?time$",
        r"^ts$",
        r"^reading_?time$",
        r"^date$",
    ],
    "asset_id": [
        r"^asset(_?id)?$",
        r"^meter(_?id|_?name)?$",
        r"^device(_?id)?$",
        r"^sensor(_?id)?$",
        r"^equipment(_?id)?$",
        r"^node(_?id)?$",
        r"^asset$",
        r"^meter$",
        r"^device$",
    ],
    "power_kw": [
        r"^(active_?)?power(_?kw)?$",
        r"^power(_?kw)?$",
        r"^kw$",
        r"^current_?kw$",
        r"^real_?power$",
        r"^load(_?kw)?$",
        r"^demand(_?kw)?$",
        r"^(active_?)?power_?w$",
        r"^watts?$",
        r"^activepower_?w$",
        r"^(active_?)?power_?mw$",
        r"^mw$",
    ],
    "interval_energy_kwh": [
        r"^interval_?energy(_?kwh)?$",
        r"^energy(_?kwh)?$",
        r"^consumption(_?kwh)?$",
        r"^kwh$",
        r"^energy_?wh$",
        r"^energy_?j(oules?)?$",
    ],
    "cumulative_energy_kwh": [
        r"^cumulative(_?energy)?(_?kwh)?$",
        r"^total(_?energy)?(_?kwh)?$",
        r"^meter_?reading(_?kwh)?$",
    ],
    "voltage_v": [
        r"^volt(age)?(_?v)?$",
        r"^v_?rms$",
        r"^volts?$",
        r"^v$",
    ],
    "current_a": [
        r"^current(_?a)?$",
        r"^amps?$",
        r"^amperes?$",
        r"^i$",
    ],
    "power_factor": [
        r"^power_?factor$",
        r"^pf$",
        r"^cos_?phi$",
        r"^powerfactor$",
    ],
    "flow_lpm": [
        r"^flow(_?lpm|_?rate)?$",
        r"^water_?flow$",
    ],
    "temperature_c": [
        r"^temp(erature)?(_?c)?$",
        r"^temp$",
    ],
}


def detect_column_mapping(columns: list[str] | pd.Index) -> dict[str, str]:
    """Auto-detect source column to canonical target mapping using heuristics.

    Returns:
        Dictionary mapping {source_column_name: canonical_target_name}.
    """
    mapping: dict[str, str] = {}
    assigned_targets: set[str] = set()

    for col in columns:
        clean = col.strip()
        norm = re.sub(r"[_\s\-]+", "_", clean.lower())

        matched_target = None
        for target, patterns in COLUMN_PATTERNS.items():
            if target in assigned_targets:
                continue
            for pat in patterns:
                if re.match(pat, norm, re.IGNORECASE):
                    matched_target = target
                    break
            if matched_target:
                break

        # Fallback substring heuristics if exact regex didn't hit
        if not matched_target:
            if "time" in norm or "date" in norm or "recorded" in norm:
                if "timestamp" not in assigned_targets:
                    matched_target = "timestamp"
            elif any(k in norm for k in ["power", "watt", "load", "kw"]):
                if "power_kw" not in assigned_targets:
                    matched_target = "power_kw"
            elif any(k in norm for k in ["meter", "device", "sensor", "asset"]):
                if "asset_id" not in assigned_targets:
                    matched_target = "asset_id"
            elif "volt" in norm:
                if "voltage_v" not in assigned_targets:
                    matched_target = "voltage_v"
            elif "curr" in norm or "amp" in norm:
                if "current_a" not in assigned_targets:
                    matched_target = "current_a"
            elif "pf" in norm or "factor" in norm:
                if "power_factor" not in assigned_targets:
                    matched_target = "power_factor"

        if matched_target:
            mapping[col] = matched_target
            assigned_targets.add(matched_target)

    return mapping


def detect_unit_conversions(source_col: str, target_col: str) -> float:
    """Infer multiplier to convert source column into canonical unit."""
    col_lower = source_col.lower()

    if target_col == "power_kw":
        # Watts -> kW
        if re.search(r"(watts?|_w$|^w$)", col_lower):
            return 0.001
        # MW -> kW
        if re.search(r"(_mw$|^mw$)", col_lower):
            return 1000.0
        return 1.0

    if target_col == "interval_energy_kwh":
        # Joules -> kWh
        if re.search(r"(joules?|_j$|^j$)", col_lower):
            return 1.0 / 3_600_000.0
        # Wh -> kWh
        if re.search(r"(_wh$|^wh$)", col_lower):
            return 0.001
        return 1.0

    return 1.0


def normalize_timezones(series: pd.Series, target_tz: str | None = None) -> pd.Series:
    """Normalize datetime series to target timezone."""
    # Coerce to datetime
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce")

    # Check for unparseable dates
    invalid_mask = parsed.isna() & series.notna()
    if invalid_mask.any():
        sample = series[invalid_mask].iloc[0]
        raise DataIngestionError(f"Unparseable datetime strings detected in timestamp column: '{sample}'")

    if target_tz is None:
        return parsed

    tz_clean = target_tz.strip()
    if tz_clean.lower() == "naive":
        if parsed.dt.tz is not None:
            return parsed.dt.tz_localize(None)
        return parsed

    alias_map = {"IST": "Asia/Kolkata", "UTC": "UTC"}
    resolved_tz = alias_map.get(tz_clean.upper(), tz_clean)

    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(resolved_tz)
    else:
        return parsed.dt.tz_convert(resolved_tz)


def process_uploaded_csv(
    file_or_path: str | Path | io.StringIO | io.BytesIO | Any,
    column_mapping: dict[str, str] | None = None,
    unit_conversions: dict[str, float | str] | None = None,
    target_tz: str | None = None,
    validate_schema: bool = True,
    expected_interval_minutes: int | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Ingest, map, convert, validate, and evaluate third-party telemetry CSV.

    Args:
        file_or_path: Path, string CSV content, or file-like object.
        column_mapping: Optional explicit mapping dict {source_col: canonical_col}
            or {canonical_col: source_col}.
        unit_conversions: Optional dict of multipliers or unit string codes.
        target_tz: Timezone string ('UTC', 'Asia/Kolkata', 'naive') or None.
        validate_schema: Whether to validate records against TelemetryReading schema.
        expected_interval_minutes: Expected sampling interval in minutes.

    Returns:
        Tuple of (canonical_telemetry_dataframe, data_quality_report).

    Raises:
        DataIngestionError: If the CSV is empty, malformed, or missing required fields.
    """
    # -------------------------------------------------------------
    # 1. Load CSV
    # -------------------------------------------------------------
    try:
        if isinstance(file_or_path, (str, Path)):
            p = Path(file_or_path)
            if p.exists() and p.is_file():
                df = pd.read_csv(p)
            else:
                # Could be raw CSV text
                raw_text = str(file_or_path).strip()
                if not raw_text:
                    raise DataIngestionError("Uploaded CSV content is empty.")
                df = pd.read_csv(io.StringIO(raw_text))
        elif hasattr(file_or_path, "read"):
            # File-like (BytesIO, StringIO, or Streamlit UploadedFile)
            content = file_or_path.read()
            if isinstance(content, bytes):
                if not content.strip():
                    raise DataIngestionError("Uploaded CSV file is empty.")
                df = pd.read_csv(io.BytesIO(content))
            else:
                if not content.strip():
                    raise DataIngestionError("Uploaded CSV file is empty.")
                df = pd.read_csv(io.StringIO(content))
        else:
            raise DataIngestionError(f"Unsupported file_or_path type: {type(file_or_path)}")
    except pd.errors.EmptyDataError:
        raise DataIngestionError("Uploaded CSV file contains no data or header.")
    except DataIngestionError:
        raise
    except Exception as e:
        raise DataIngestionError(f"Failed to read CSV input: {e}") from e

    if df.empty:
        raise DataIngestionError("Uploaded CSV contains 0 data rows.")

    # -------------------------------------------------------------
    # 2. Build Column Mapping (Smart Auto-detection + User Overrides)
    # -------------------------------------------------------------
    auto_mapping = detect_column_mapping(df.columns)
    effective_mapping = dict(auto_mapping)

    if column_mapping:
        # Normalize mapping orientation: support both {src: target} and {target: src}
        for k, v in column_mapping.items():
            if k in df.columns:
                effective_mapping[k] = v
            elif v in df.columns:
                effective_mapping[v] = k

    # Invert mapping to find source column for each canonical target
    target_to_source: dict[str, str] = {}
    for src, tgt in effective_mapping.items():
        if tgt not in target_to_source and src in df.columns:
            target_to_source[tgt] = src

    # Verify required canonical fields
    if "timestamp" not in target_to_source:
        raise DataIngestionError(
            "Missing required column 'timestamp'. "
            f"Available headers: {list(df.columns)}. "
            "Please provide a mapping to a valid timestamp column."
        )

    if "power_kw" not in target_to_source:
        raise DataIngestionError(
            "Missing required power column (e.g. 'power_kw', 'kW', 'Power', 'ActivePower_W'). "
            f"Available headers: {list(df.columns)}. "
            "Please provide a mapping to a power column."
        )

    # -------------------------------------------------------------
    # 3. Create Canonical Dataframe
    # -------------------------------------------------------------
    canonical_df = pd.DataFrame(index=df.index)

    # Map timestamp
    src_ts = target_to_source["timestamp"]
    canonical_df["timestamp"] = normalize_timezones(df[src_ts], target_tz=target_tz)

    # Map asset_id
    if "asset_id" in target_to_source:
        canonical_df["asset_id"] = df[target_to_source["asset_id"]].astype(str)
    else:
        canonical_df["asset_id"] = "EXTERNAL_METER_01"

    # Map power_kw with unit conversions
    src_pwr = target_to_source["power_kw"]
    pwr_series = pd.to_numeric(df[src_pwr], errors="coerce")
    if pwr_series.isna().any() and df[src_pwr].notna().any():
        invalid_sample = df[src_pwr][pwr_series.isna() & df[src_pwr].notna()].iloc[0]
        raise DataIngestionError(f"Non-numeric values found in power column '{src_pwr}': '{invalid_sample}'")

    multiplier = 1.0
    if unit_conversions and src_pwr in unit_conversions:
        uc = unit_conversions[src_pwr]
        if isinstance(uc, (int, float)):
            multiplier = float(uc)
        elif isinstance(uc, str):
            code = uc.lower().strip()
            if code in ["w->kw", "w_to_kw", "watts_to_kw"]:
                multiplier = 0.001
            elif code in ["mw->kw", "mw_to_kw"]:
                multiplier = 1000.0
    else:
        multiplier = detect_unit_conversions(src_pwr, "power_kw")

    canonical_df["power_kw"] = pwr_series * multiplier

    # Map other optional columns
    optional_targets = [
        "interval_energy_kwh",
        "cumulative_energy_kwh",
        "voltage_v",
        "current_a",
        "power_factor",
        "flow_lpm",
        "temperature_c",
    ]

    for tgt in optional_targets:
        if tgt in target_to_source:
            src_col = target_to_source[tgt]
            s = pd.to_numeric(df[src_col], errors="coerce")
            tgt_mult = 1.0
            if unit_conversions and src_col in unit_conversions:
                uc = unit_conversions[src_col]
                tgt_mult = float(uc) if isinstance(uc, (int, float)) else 1.0
            else:
                tgt_mult = detect_unit_conversions(src_col, tgt)
            canonical_df[tgt] = s * tgt_mult
        else:
            canonical_df[tgt] = None

    # Populate platform identifiers and statuses
    canonical_df["reading_id"] = [f"rec_{uuid.uuid4().hex[:8]}" for _ in range(len(canonical_df))]
    canonical_df["reading_type"] = "meter"
    canonical_df["sensor_status"] = SensorStatus.HEALTHY.value
    canonical_df["quality_flag"] = QualityFlag.VALID.value

    # -------------------------------------------------------------
    # 4. Evaluate Telemetry Quality
    # -------------------------------------------------------------
    engine = DataQualityEngine(expected_interval_minutes=expected_interval_minutes)
    tagged_df, quality_report = engine.evaluate(canonical_df)

    # -------------------------------------------------------------
    # 5. Schema Contract Validation
    # -------------------------------------------------------------
    if validate_schema:
        # Validate sample or all rows against TelemetryReading model
        sample_rows = tagged_df.head(min(100, len(tagged_df)))
        for idx, row in sample_rows.iterrows():
            row_dict = row.to_dict()
            try:
                TelemetryReading.model_validate(row_dict)
            except ValidationError as ve:
                raise DataIngestionError(f"Row {idx} fails TelemetryReading validation: {ve}") from ve

    return tagged_df, quality_report
