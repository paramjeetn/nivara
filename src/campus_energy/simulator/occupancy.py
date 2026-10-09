"""Occupancy model for the campus simulator.

Generates realistic occupancy counts based on:
- Zone capacity
- Schedule (working days, time slots, occupancy fractions)
- Holidays
- Small random variation (not pure noise — structured around schedules)

Occupancy is always: 0 <= occupancy <= zone_capacity
"""
import numpy as np
from datetime import datetime

from campus_energy.config import SimulationConfig, ZoneConfig
from campus_energy.simulator.schedules import (
    get_scheduled_occupancy_fraction,
    is_holiday,
)


def generate_occupancy(
    dt: datetime,
    zone: ZoneConfig,
    schedule_id: str,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Generate (actual_occupancy, scheduled_occupancy) for a zone at a timestamp.
    
    Returns:
        Tuple of (actual_occupancy, scheduled_occupancy)
        Both are integers >= 0 and <= zone.capacity
    """
    scheduled_fraction = get_scheduled_occupancy_fraction(dt, schedule_id, config)
    scheduled_occupancy = int(round(zone.capacity * scheduled_fraction))
    
    if scheduled_fraction <= 0.0:
        # Outside scheduled hours: small chance of stragglers
        # Different behavior by zone type
        if zone.type in ("common_area",):
            # Hostel common areas can have some people even outside "peak" times
            straggler_fraction = rng.uniform(0.02, 0.08)
            actual = int(round(zone.capacity * straggler_fraction))
        elif zone.type in ("reading_hall",):
            # Library might have a few people outside slots on working days
            day_of_week = dt.isoweekday()
            if day_of_week <= 6 and not is_holiday(dt, config):
                hour = dt.hour
                if 7 <= hour <= 22:
                    straggler_fraction = rng.uniform(0.02, 0.10)
                    actual = int(round(zone.capacity * straggler_fraction))
                else:
                    actual = 0
            else:
                actual = 0
        elif zone.type == "pump_room":
            actual = 0  # Pump rooms don't have occupancy in the people sense
        else:
            # Most zones: empty outside schedule
            actual = 0
    else:
        # During scheduled hours: vary around the scheduled fraction
        # Use a bounded normal distribution
        noise_std = 0.10  # 10% variation around scheduled
        noise = rng.normal(0, noise_std)
        actual_fraction = scheduled_fraction + noise
        
        # Add inter-class transition dips for classrooms
        if zone.type in ("classroom", "computer_lab", "laboratory"):
            minute = dt.minute
            # At the top of each hour (transition periods), reduce occupancy
            if minute < 5 or minute >= 55:
                actual_fraction *= rng.uniform(0.3, 0.6)
        
        actual_fraction = np.clip(actual_fraction, 0.0, 1.0)
        actual = int(round(zone.capacity * actual_fraction))
    
    # Enforce bounds
    actual = max(0, min(actual, zone.capacity))
    scheduled_occupancy = max(0, min(scheduled_occupancy, zone.capacity))
    
    return actual, scheduled_occupancy
