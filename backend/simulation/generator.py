"""Sensor data simulation / generation."""

import math
import random

from core.sensors import Sensor, SensorKind


def simulate_tick(
    sensors_by_room: dict[str, list[Sensor]],
    tick: int,
) -> None:
    """Update sensor values in-place for the given tick.

    Generates realistic data with:
    - Time-of-day patterns (tick maps to time progression)
    - Gradual drift (not random noise each tick)
    - Occasional waste scenarios
    """
    hour = (tick * 0.05) % 24  # each tick ~ 3 minutes of simulated time

    for room_id, sensors in sensors_by_room.items():
        # Determine floor from the first sensor in the room
        floor = sensors[0].floor if sensors else 0
        is_occupied = _is_room_occupied(room_id, hour, tick, floor)
        for sensor in sensors:
            match sensor.kind:
                case SensorKind.TEMPERATURE:
                    sensor.value = _sim_temperature(room_id, hour, tick, is_occupied)
                case SensorKind.OCCUPANCY:
                    sensor.value = 1.0 if is_occupied else 0.0
                case SensorKind.CO2:
                    sensor.value = _sim_co2(room_id, hour, tick, is_occupied)
                case SensorKind.POWER:
                    sensor.value = _sim_power(room_id, hour, tick, is_occupied)
                case SensorKind.LIGHT:
                    sensor.value = 1.0 if is_occupied else (0.2 if _is_waste_scenario(room_id, tick) else 0.0)


def _is_room_occupied(room_id: str, hour: float, tick: int, floor: int) -> bool:
    """Simulate occupancy patterns. More occupied during day hours."""
    room_hash = hash(room_id) % 100
    is_upper_floor = floor > 0

    if 8 <= hour < 22:  # daytime
        base_prob = 0.6
    elif hour >= 22 or hour < 6:  # nighttime — upper floors are bedrooms
        base_prob = 0.8 if is_upper_floor else 0.1
    else:  # early morning
        base_prob = 0.3

    noise = math.sin(tick * 0.1 + room_hash) * 0.2
    return random.Random(tick * 100 + room_hash).random() < (base_prob + noise)


def _sim_temperature(room_id: str, hour: float, tick: int, occupied: bool) -> float:
    """Simulate temperature with drift, 18-26C range."""
    base = 21.0
    base += math.sin((hour - 6) * math.pi / 12) * 1.5  # warmer midday
    if occupied:
        base += 0.5
    if _is_waste_scenario(room_id, tick):
        base += 3.0
    base += math.sin(tick * 0.3 + hash(room_id)) * 0.5
    return round(max(16.0, min(28.0, base)), 1)


def _sim_co2(room_id: str, hour: float, tick: int, occupied: bool) -> float:
    """Simulate CO2 levels. 400 baseline, rises with occupancy."""
    _ = hour  # hour unused but kept for API consistency
    base = 400.0
    if occupied:
        base += 150.0 + math.sin(tick * 0.2 + hash(room_id)) * 50
    else:
        base += math.sin(tick * 0.1) * 20
    return round(max(350.0, min(1200.0, base)), 0)


def _sim_power(room_id: str, hour: float, tick: int, occupied: bool) -> float:
    """Simulate power draw in watts."""
    base = 30.0  # standby power
    if occupied:
        base += 150.0
    base += 80.0 + math.sin((hour - 6) * math.pi / 12) * 40
    if not occupied and _is_waste_scenario(room_id, tick):
        base += 120.0
    base += math.sin(tick * 0.5 + hash(room_id)) * 15
    return round(max(10.0, base), 0)


def _is_waste_scenario(room_id: str, tick: int) -> bool:
    """Periodically inject waste scenarios for certain rooms."""
    room_hash = hash(room_id) % 7
    cycle = (tick + room_hash * 5) % 40
    return cycle < 12
