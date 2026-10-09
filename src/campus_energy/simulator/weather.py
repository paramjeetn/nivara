"""Weather model for the campus simulator.

Generates outdoor temperature and humidity using daily sinusoidal cycle,
multi-day slow variation, and bounded Gaussian noise.

All temperatures in Celsius. Humidity in percent (0-100).
"""
import numpy as np
from datetime import datetime

from campus_energy.config import WeatherConfig


def generate_weather(
    dt: datetime,
    weather_config: WeatherConfig,
    simulation_start: datetime,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Generate (temperature_c, humidity_pct) for a given timestamp.

    Model:
    - Base temp + daily sinusoidal cycle peaking at peak_hour
    - Multi-day sinusoidal variation
    - Gaussian noise
    - Temperature bounded to [18, 45] C
    - Humidity bounded to [30, 98] %
    """
    hours_elapsed = (dt - simulation_start).total_seconds() / 3600.0
    hour_of_day = dt.hour + dt.minute / 60.0

    # Daily cycle: peaks at peak_hour
    daily_phase = 2 * np.pi * (hour_of_day - weather_config.peak_hour) / 24.0
    daily_component = weather_config.daily_amplitude_c * np.cos(daily_phase)

    # Multi-day slow variation
    days_elapsed = hours_elapsed / 24.0
    multi_day_phase = 2 * np.pi * days_elapsed / weather_config.multi_day_period_days
    multi_day_component = weather_config.multi_day_amplitude_c * np.sin(multi_day_phase)

    # Temperature with noise
    temperature = (
        weather_config.base_temperature_c
        + daily_component
        + multi_day_component
        + rng.normal(0, weather_config.noise_std_temperature_c)
    )
    temperature = float(np.clip(temperature, 18.0, 45.0))

    # Humidity: inversely correlated with temperature
    humidity_daily = -weather_config.humidity_amplitude_pct * np.cos(daily_phase)
    humidity = (
        weather_config.base_humidity_pct
        + humidity_daily
        + rng.normal(0, weather_config.noise_std_humidity_pct)
    )
    humidity = float(np.clip(humidity, 30.0, 98.0))

    return round(temperature, 2), round(humidity, 2)
