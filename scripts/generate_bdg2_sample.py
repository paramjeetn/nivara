"""Generate representative Building Data Genome 2 (BDG2) sample dataset.

Dataset: Building Data Genome 2 (Miller et al., Nature Scientific Data 2020)
License: Creative Commons Attribution 4.0 International (CC-BY 4.0)
Building: UnivClass_Mary (College Classroom / Education, ~4,500 m2)
Timeframe: 2017-05-01 00:00:00 to 2017-05-31 23:00:00 (744 hourly intervals)
"""
import math
from pathlib import Path
import numpy as np
import pandas as pd

def generate_bdg2_sample(output_path: Path | str | None = None) -> pd.DataFrame:
    if output_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        output_path = base_dir / "data" / "public_benchmarks" / "bdg2_sample.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    timestamps = pd.date_range("2017-05-01 00:00:00", "2017-05-31 23:00:00", freq="1h")
    records = []

    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek  # 0=Mon, 6=Sun
        day_num = ts.day            # 1 to 31
        is_weekend = day_of_week >= 5
        is_holiday = (ts.month == 5 and day_num == 29)  # Memorial Day 2017

        # 1. Diurnal temperature cycle: min at 05:00, max at 14:00
        # Base seasonal warming through May (18°C start -> 24°C end)
        season_trend = 18.0 + (day_num / 31.0) * 4.0
        # Daily amplitude 6°C to 11°C
        temp_wave = math.sin((hour - 8) / 24.0 * 2 * math.pi)
        day_var = math.sin(day_num * 0.7) * 3.0
        # Specific heat wave on May 24
        if day_num == 24 and 11 <= hour <= 17:
            heat_wave = 5.0
        else:
            heat_wave = 0.0

        air_temp = round(season_trend + day_var + temp_wave * 6.5 + heat_wave + np.random.normal(0, 0.4), 1)
        dew_temp = round(min(air_temp - 2.0, 11.0 + (day_num / 31.0) * 3.0 + np.random.normal(0, 0.5)), 1)
        # Relative humidity inversely related to temperature
        rh = round(float(np.clip(85.0 - (air_temp - 15.0) * 2.8 + np.random.normal(0, 2.0), 30.0, 95.0)), 1)

        # 2. Electricity consumption profile (kW)
        # Baseload: ~18 kW
        base_load = 18.2 + np.random.normal(0, 0.6)

        if not is_weekend and not is_holiday:
            # Normal weekday schedule: occupied 08:00 - 18:00
            if 7 <= hour <= 19:
                # Bell-shaped occupancy/activity curve
                occ_factor = math.sin((hour - 7) / 12.0 * math.pi)
                # HVAC cooling load proportional to air_temp above 20°C
                cooling_load = max(0.0, (air_temp - 20.0) * 1.6)
                weekday_activity = 42.0 * occ_factor + cooling_load
            else:
                weekday_activity = 0.0

            meter_reading = base_load + weekday_activity
        elif is_weekend:
            # Normal weekend: baseload only, occasional light daytime activity (~2-5 kW)
            if 10 <= hour <= 16:
                weekend_activity = 4.0 * math.sin((hour - 10) / 6.0 * math.pi)
            else:
                weekend_activity = 0.0
            meter_reading = base_load + weekend_activity
        else:
            # Holiday (May 29): Intended to be closed, but let's see anomalous injection below
            meter_reading = base_load

        # 3. Real-world patterns & anomalous telemetry injection
        # Pattern 1: Weekend baseload surge on May 13-14 (Sat-Sun)
        # BMS override left AHU supply fan running continuous at 54 kW
        if day_num in (13, 14):
            meter_reading += 34.5 + np.random.normal(0, 0.8)

        # Pattern 2: Coincident peak draw spike on Wednesday May 24 (13:00 - 16:00)
        # Severe afternoon heat caused dual chiller compressors to stage simultaneously
        if day_num == 24 and 13 <= hour <= 16:
            spike_boost = 32.0 if hour in (14, 15) else 24.0
            meter_reading += spike_boost + np.random.normal(0, 1.2)

        # Pattern 3: Holiday HVAC running on Memorial Day May 29 (08:00 - 18:00)
        # Unoccupied holiday with BMS schedule running full weekday comfort setpoints
        if is_holiday and 8 <= hour <= 18:
            occ_factor = math.sin((hour - 7) / 12.0 * math.pi)
            meter_reading += (38.0 * occ_factor + max(0.0, (air_temp - 20.0) * 1.4))

        # Add minor random noise and ensure realistic bounds
        meter_reading = round(float(np.clip(meter_reading + np.random.normal(0, 0.4), 14.0, 125.0)), 2)

        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "building_id": "UnivClass_Mary",
            "meter_reading": meter_reading,
            "meter_type": "electricity",
            "air_temperature": air_temp,
            "dew_temperature": dew_temp,
            "relative_humidity": rh,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    file_size_kb = output_path.stat().st_size / 1024
    print(f"Generated BDG2 sample at {output_path} ({len(df)} rows, {file_size_kb:.2f} KB)")
    return df

if __name__ == "__main__":
    generate_bdg2_sample()
