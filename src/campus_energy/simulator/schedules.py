"""Operating schedule logic for the campus simulator."""
from datetime import datetime, time
from typing import Optional

from campus_energy.config import SimulationConfig, Schedule


def is_holiday(dt: datetime, config: SimulationConfig) -> bool:
    """Check if the given date is a configured holiday."""
    date_str = dt.strftime("%Y-%m-%d")
    return any(h.date == date_str for h in config.holidays)


def get_schedule(schedule_id: str, config: SimulationConfig) -> Schedule:
    """Get schedule definition by ID."""
    if schedule_id not in config.schedules:
        raise ValueError(f"Unknown schedule: {schedule_id}")
    return config.schedules[schedule_id]


def is_working_time(dt: datetime, schedule_id: str, config: SimulationConfig) -> bool:
    """Check if the given datetime falls within working hours for the schedule.

    Returns False on holidays for non-continuous schedules (academic, office, meeting).
    Hostel and pump schedules operate regardless of holidays.
    """
    schedule = get_schedule(schedule_id, config)
    day_of_week = dt.isoweekday()
    if day_of_week not in schedule.working_days:
        return False
    continuous_schedules = {"SCH-HOSTEL-COMMON", "SCH-HOSTEL-UTILITY", "SCH-PUMP"}
    if schedule_id not in continuous_schedules and is_holiday(dt, config):
        return False
    current_time = dt.time()
    for slot in schedule.slots:
        start = time.fromisoformat(slot.start)
        end = time.fromisoformat(slot.end)
        if start <= current_time < end:
            return True
    return False


def get_scheduled_occupancy_fraction(
    dt: datetime, schedule_id: str, config: SimulationConfig
) -> float:
    """Get the scheduled occupancy fraction for a given datetime.

    Returns 0.0 if outside working hours, on non-working days, or on holidays
    (for non-continuous schedules).
    """
    schedule = get_schedule(schedule_id, config)
    day_of_week = dt.isoweekday()
    if day_of_week not in schedule.working_days:
        return 0.0
    continuous_schedules = {"SCH-HOSTEL-COMMON", "SCH-HOSTEL-UTILITY", "SCH-PUMP"}
    if schedule_id not in continuous_schedules and is_holiday(dt, config):
        return 0.0
    current_time = dt.time()
    for slot in schedule.slots:
        start = time.fromisoformat(slot.start)
        end = time.fromisoformat(slot.end)
        if start <= current_time < end:
            return slot.occupancy_fraction
    return 0.0


def get_activity_type(
    zone_type: str, schedule_id: str, dt: datetime, config: SimulationConfig
) -> Optional[str]:
    """Determine the academic activity type for a zone at a given time."""
    if not is_working_time(dt, schedule_id, config):
        return None
    activity_map = {
        "classroom": "lecture",
        "computer_lab": "lab",
        "laboratory": "lab",
        "office": "office_work",
        "meeting_room": "meeting",
        "reading_hall": "study",
        "common_area": "recreation",
        "pump_room": "operations",
        "utility": "maintenance",
    }
    return activity_map.get(zone_type)
