"""Sensor types and readings."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.environment import RoomEnvironmentSource


class SensorKind(StrEnum):
    TEMPERATURE = "temperature"
    OCCUPANCY = "occupancy"
    CO2 = "co2"
    HEATING_POWER = "heating_power"
    VENTILATION_POWER = "ventilation_power"


@dataclass
class Sensor:
    """A sensor placed in a room.

    Sensors don't store values - they read from an environment source.
    This allows the same sensor to work with real or simulated data.
    """

    id: str
    kind: SensorKind
    room_id: str
    x: float
    y: float
    floor: int

    def read(self, source: RoomEnvironmentSource) -> float:
        """Read current value from the environment source."""
        state = source.get_state(self.room_id)
        match self.kind:
            case SensorKind.TEMPERATURE:
                return round(state.temperature, 1)
            case SensorKind.OCCUPANCY:
                return 1.0 if state.occupied else 0.0
            case SensorKind.CO2:
                return round(state.co2_ppm, 0)
            case SensorKind.HEATING_POWER:
                return round(state.heating_power_w, 0)
            case SensorKind.VENTILATION_POWER:
                return round(state.ventilation_power_w, 0)


@dataclass
class SensorReading:
    sensor_id: str
    kind: SensorKind
    value: float
    tick: int
