from dataclasses import dataclass
from enum import StrEnum


class SensorKind(StrEnum):
    TEMPERATURE = "temperature"
    OCCUPANCY = "occupancy"
    CO2 = "co2"
    POWER = "power"
    LIGHT = "light"


@dataclass
class Sensor:
    id: str
    kind: SensorKind
    x: float
    y: float
    floor: int
    value: float = 0.0  # current reading


@dataclass
class SensorReading:
    sensor_id: str
    kind: SensorKind
    value: float
    tick: int
