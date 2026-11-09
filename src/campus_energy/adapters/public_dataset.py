"""Public dataset adapter for Building Data Genome 2 (BDG2).

Provides ingestion, schema normalization, and data quality validation for
real-world commercial building energy telemetry published under Creative
Commons Attribution 4.0 International (CC-BY 4.0).

Reference:
    Miller, C., Kathirgamanathan, A., Picchetti, B., Arjunan, P., Park, J. Y.,
    Nagy, Z., ... & Schlueter, A. (2020). The Building Data Genome 2 (BDG2)
    Data Set: open hourly electricity, water, gas, and steam meter readings
    for 3,053 buildings. Scientific Data, 7(1), 1-14.
    License: CC-BY 4.0 (https://creativecommons.org/licenses/by/4.0/)
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from campus_energy.adapters.quality import DataQualityEngine
from campus_energy.schemas import (
    Asset,
    AssetType,
    ContextRecord,
    DataQualityReport,
    EquipmentState,
    MonitoringLevel,
    QualityFlag,
    SensorStatus,
    TelemetryReading,
)

logger = logging.getLogger(__name__)

# Official BDG2 metadata and open-science attribution
BDG2_METADATA = {
    "dataset_name": "Building Data Genome 2 (BDG2)",
    "publication": "Nature Scientific Data 2020",
    "authors": "Clayton Miller et al.",
    "license": "Creative Commons Attribution 4.0 International (CC-BY 4.0)",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "doi": "10.1038/s41597-020-00712-x",
    "provenance": "Open building energy data repository from 3,053 global commercial/higher education facilities",
}


class BDG2Adapter:
    """Adapter for ingesting and validating Building Data Genome 2 datasets into Nivara."""

    def __init__(
        self,
        filepath: Path | str | None = None,
        expected_interval_minutes: int = 60,
    ) -> None:
        """Initialize BDG2Adapter.

        Args:
            filepath: Path to raw BDG2 CSV. If None, defaults to data/public_benchmarks/bdg2_sample.csv.
            expected_interval_minutes: Expected sampling resolution in minutes (default 60 for BDG2).
        """
        if filepath is None:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            self.filepath = base_dir / "data" / "public_benchmarks" / "bdg2_sample.csv"
        else:
            self.filepath = Path(filepath)

        self.expected_interval_minutes = expected_interval_minutes
        self.quality_engine = DataQualityEngine(
            expected_interval_minutes=expected_interval_minutes,
            stuck_threshold_steps=12,
            outlier_power_max_kw=10000.0,
            outlier_power_min_kw=0.0,
        )

    def load_raw(self, filepath: Path | str | None = None) -> pd.DataFrame:
        """Load raw BDG2 CSV file into a pandas DataFrame.

        Verifies presence of core BDG2 columns:
        timestamp, building_id, meter_reading, meter_type.
        """
        target = Path(filepath) if filepath is not None else self.filepath
        if not target.exists():
            raise FileNotFoundError(f"BDG2 dataset file not found at: {target}")

        df = pd.read_csv(target)
        required_cols = {"timestamp", "building_id", "meter_reading", "meter_type"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"BDG2 CSV missing required column(s): {missing_cols}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        return df

    def to_telemetry_and_context(
        self,
        raw_df: pd.DataFrame | None = None,
    ) -> tuple[list[TelemetryReading], list[ContextRecord], DataQualityReport]:
        """Convert raw BDG2 dataset into typed TelemetryReading and ContextRecord schemas.

        Runs DataQualityEngine verification on telemetry before instantiating schemas.

        Returns:
            Tuple of (telemetry_readings, context_records, quality_report).
        """
        if raw_df is None:
            raw_df = self.load_raw()

        telemetry_df, context_df, _, quality_report = self.to_dataframes(raw_df)

        # Convert telemetry_df to TelemetryReading Pydantic instances
        telemetry_records: list[TelemetryReading] = []
        for _, row in telemetry_df.iterrows():
            q_flag = QualityFlag(row.get("quality_flag", QualityFlag.VALID.value))
            s_status = SensorStatus(row.get("sensor_status", SensorStatus.HEALTHY.value))
            telemetry_records.append(
                TelemetryReading(
                    reading_id=str(row["reading_id"]),
                    timestamp=pd.to_datetime(row["timestamp"]).to_pydatetime(),
                    asset_id=str(row["asset_id"]),
                    reading_type="meter",
                    quality_flag=q_flag,
                    sensor_status=s_status,
                    power_kw=float(row["power_kw"]) if pd.notna(row.get("power_kw")) else None,
                    interval_energy_kwh=(
                        float(row["interval_energy_kwh"])
                        if pd.notna(row.get("interval_energy_kwh"))
                        else None
                    ),
                    cumulative_energy_kwh=(
                        float(row["cumulative_energy_kwh"])
                        if pd.notna(row.get("cumulative_energy_kwh"))
                        else None
                    ),
                    voltage_v=float(row["voltage_v"]) if pd.notna(row.get("voltage_v")) else None,
                    power_factor=float(row["power_factor"]) if pd.notna(row.get("power_factor")) else None,
                )
            )

        # Convert context_df to ContextRecord Pydantic instances
        context_records: list[ContextRecord] = []
        for _, row in context_df.iterrows():
            context_records.append(
                ContextRecord(
                    timestamp=pd.to_datetime(row["timestamp"]).to_pydatetime(),
                    building_id=str(row["building_id"]),
                    zone_id=str(row["zone_id"]) if pd.notna(row.get("zone_id")) else None,
                    occupancy=int(row["occupancy"]) if pd.notna(row.get("occupancy")) else 0,
                    scheduled_occupancy=(
                        int(row["scheduled_occupancy"])
                        if pd.notna(row.get("scheduled_occupancy"))
                        else 0
                    ),
                    occupancy_source=str(row.get("occupancy_source", "bdg2_proxy")),
                    working_hours=bool(row.get("working_hours", False)),
                    holiday=bool(row.get("holiday", False)),
                    academic_activity=(
                        str(row["academic_activity"])
                        if pd.notna(row.get("academic_activity"))
                        else None
                    ),
                    day_of_week=int(row["day_of_week"]),
                    outdoor_temperature_c=float(row["outdoor_temperature_c"]),
                    humidity_pct=float(np.clip(float(row.get("humidity_pct", 50.0)), 0.0, 100.0)),
                )
            )

        return telemetry_records, context_records, quality_report

    def to_dataframes(
        self,
        raw_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DataQualityReport]:
        """Convert raw BDG2 dataset into normalized telemetry, context, and asset DataFrames.

        Runs DataQualityEngine verification and returns:
            (telemetry_df, context_df, assets_df, quality_report)
        """
        if raw_df is None:
            raw_df = self.load_raw()

        df = raw_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

        primary_building = str(df["building_id"].iloc[0]) if not df.empty else "UnivClass_Mary"
        meter_asset_id = f"MTR-{primary_building}"

        # 1. Telemetry DataFrame Preparation
        # In BDG2, meter_reading for electricity is in kWh per hourly interval,
        # which equals average active power in kW.
        power_vals = pd.to_numeric(df["meter_reading"], errors="coerce")
        interval_hours = self.expected_interval_minutes / 60.0
        interval_kwh = power_vals * interval_hours

        initial_tel = pd.DataFrame({
            "timestamp": df["timestamp"],
            "asset_id": meter_asset_id,
            "reading_type": "meter",
            "power_kw": power_vals,
            "interval_energy_kwh": interval_kwh,
            "voltage_v": 230.0,
            "power_factor": 0.92,
        })

        # Run DataQualityEngine
        tagged_tel, quality_report = self.quality_engine.evaluate(initial_tel)

        # Assign unique reading IDs and compute cumulative energy
        tagged_tel["reading_id"] = [
            f"TEL-BDG2-{primary_building}-{i+1:05d}" for i in range(len(tagged_tel))
        ]
        tagged_tel["cumulative_energy_kwh"] = (
            tagged_tel["interval_energy_kwh"].fillna(0).cumsum()
        )

        # Reorder telemetry columns
        tel_cols = [
            "reading_id",
            "timestamp",
            "asset_id",
            "reading_type",
            "quality_flag",
            "sensor_status",
            "power_kw",
            "interval_energy_kwh",
            "cumulative_energy_kwh",
            "voltage_v",
            "power_factor",
        ]
        telemetry_df = tagged_tel[[c for c in tel_cols if c in tagged_tel.columns]].copy()

        # 2. Context DataFrame Preparation
        # Detect weather columns (handling various naming conventions in BDG2)
        weather_col_temp = None
        for col_cand in ["air_temperature", "airTemperature", "outdoor_temperature_c", "temperature"]:
            if col_cand in df.columns:
                weather_col_temp = col_cand
                break

        weather_col_rh = None
        for col_cand in ["relative_humidity", "relativeHumidity", "humidity_pct", "humidity"]:
            if col_cand in df.columns:
                weather_col_rh = col_cand
                break

        if weather_col_temp is not None:
            outdoor_temp = pd.to_numeric(df[weather_col_temp], errors="coerce").fillna(22.0)
        else:
            outdoor_temp = pd.Series([22.0] * len(df), index=df.index)

        if weather_col_rh is not None:
            rh_pct = pd.to_numeric(df[weather_col_rh], errors="coerce").fillna(50.0)
        else:
            rh_pct = pd.Series([50.0] * len(df), index=df.index)

        day_of_week = df["timestamp"].dt.dayofweek
        hour = df["timestamp"].dt.hour

        # Memorial Day 2017: May 29, 2017
        is_holiday = (
            (df["timestamp"].dt.month == 5)
            & (df["timestamp"].dt.day == 29)
            & (df["timestamp"].dt.year == 2017)
        )
        working_hours = (hour >= 8) & (hour < 18) & (day_of_week < 5) & (~is_holiday)

        # Estimated occupancy proxy for educational building
        scheduled_occ = np.where(working_hours, 120, 0)
        proxy_occ = np.where(working_hours, np.clip((outdoor_temp - 15) * 5 + 80, 40, 150).astype(int), 0)

        context_df = pd.DataFrame({
            "timestamp": df["timestamp"],
            "building_id": primary_building,
            "zone_id": None,
            "occupancy": proxy_occ,
            "scheduled_occupancy": scheduled_occ,
            "occupancy_source": "bdg2_proxy",
            "working_hours": working_hours,
            "holiday": is_holiday,
            "academic_activity": np.where(working_hours, "lecture", "none"),
            "day_of_week": day_of_week,
            "outdoor_temperature_c": outdoor_temp,
            "humidity_pct": np.clip(rh_pct, 0.0, 100.0),
        })

        # 3. Assets DataFrame Preparation
        max_load = float(telemetry_df["power_kw"].max()) if not telemetry_df.empty else 100.0
        rated_power_kw = round(max(max_load * 1.25, 50.0), 1)

        assets_df = pd.DataFrame([
            {
                "asset_id": meter_asset_id,
                "asset_type": AssetType.METER.value,
                "asset_name": f"{primary_building} Main Electricity Meter",
                "building_id": primary_building,
                "floor": 1,
                "zone_id": None,
                "parent_asset_id": primary_building,
                "rated_power_kw": rated_power_kw,
                "equipment_count": 1,
                "operating_schedule_id": "EDU_WEEKDAY_CLASS",
                "monitoring_level": MonitoringLevel.BUILDING.value,
                "active": True,
            },
            {
                "asset_id": primary_building,
                "asset_type": AssetType.BUILDING.value,
                "asset_name": f"{primary_building} Education & Research Building",
                "building_id": primary_building,
                "floor": 4,
                "zone_id": None,
                "parent_asset_id": None,
                "rated_power_kw": rated_power_kw,
                "equipment_count": None,
                "operating_schedule_id": "EDU_WEEKDAY_CLASS",
                "monitoring_level": MonitoringLevel.BUILDING.value,
                "active": True,
            },
        ])

        return telemetry_df, context_df, assets_df, quality_report


def load_bdg2_benchmark(
    filepath: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DataQualityReport]:
    """Convenience function to load and process BDG2 benchmark dataset into DataFrames."""
    adapter = BDG2Adapter(filepath=filepath)
    return adapter.to_dataframes()
