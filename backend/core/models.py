"""Core data models for the building energy system."""

from dataclasses import dataclass, field


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
    temperature: float
    occupancy: bool
    co2: float
    power: float
    waste_patterns: list[str] = field(default_factory=list)
    heat_flow: float = 0.0  # net heat gain/loss in watts (positive = gaining heat)


@dataclass
class MetricsUpdate:
    tick: int
    rooms: dict[str, RoomMetrics]
