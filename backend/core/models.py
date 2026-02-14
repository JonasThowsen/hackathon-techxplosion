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
    """Metrics for a single room - sent via WebSocket."""

    temperature: float
    occupancy: bool
    co2: float
    heating_power: float  # Power used for heating (W)
    ventilation_power: float  # Power used for ventilation (W)
    waste_patterns: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    predicted_temp_30min: float | None = None
    predicted_temp_1h: float | None = None
    predicted_temp_2h: float | None = None
    prediction_uncertainty: float | None = None
    prediction_warnings: list[str] = field(default_factory=list)
    uses_estimated_params: bool = False

    @property
    def total_hvac_power(self) -> float:
        return self.heating_power + self.ventilation_power


@dataclass
class HeatFlow:
    """Heat transfer between two adjacent rooms."""

    from_room: str
    to_room: str
    watts: float


@dataclass
class MetricsUpdate:
    tick: int
    rooms: dict[str, RoomMetrics]
    heat_flows: list[HeatFlow] = field(default_factory=list)
    system_enabled: bool = True
    sun_enabled: bool = True
    external_temp_c: float = 0.0
    simulated_time: str = ""  # ISO format timestamp representing simulation time
