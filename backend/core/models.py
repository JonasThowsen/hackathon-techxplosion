"""Core data models for the building energy system."""

from dataclasses import dataclass


@dataclass
class Room:
    id: str
    name: str
    polygon: list[list[float]]


@dataclass
class Floor:
    floor_index: int
    label: str
    rooms: list[Room]


@dataclass
class BuildingLayout:
    id: str
    name: str
    width_m: float
    height_m: float
    floors: list[Floor]


@dataclass
class RoomMetrics:
    """Metrics for a single room - sent via WebSocket."""

    temperature: float
    occupancy: bool
    co2: float
    heating_power: float  # Power used for heating (W)
    ventilation_power: float  # Power used for ventilation (W)

    @property
    def total_hvac_power(self) -> float:
        return self.heating_power + self.ventilation_power


@dataclass
class MetricsUpdate:
    tick: int
    rooms: dict[str, RoomMetrics]
