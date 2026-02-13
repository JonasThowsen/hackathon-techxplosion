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


@dataclass
class MetricsUpdate:
    tick: int
    rooms: dict[str, RoomMetrics]
