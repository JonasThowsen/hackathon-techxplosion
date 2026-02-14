"""EnergyZone abstract base class and associated types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Metrics:
    """Aggregated metrics at any level of the zone hierarchy."""

    temperature: float
    occupancy: bool
    co2: float
    heating_power: float
    ventilation_power: float

    @property
    def total_hvac_power(self) -> float:
        return self.heating_power + self.ventilation_power


# ---------------------------------------------------------------------------
# Waste pattern variants
# ---------------------------------------------------------------------------


@dataclass
class EmptyRoomHeating:
    """Heating running while the room is unoccupied."""

    room_name: str
    estimated_kwh_wasted: float
    duration_minutes: float


@dataclass
class OverHeating:
    """Temperature exceeds the comfort threshold."""

    room_name: str
    estimated_kwh_wasted: float
    duration_minutes: float


type WastePattern = EmptyRoomHeating | OverHeating


def waste_pattern_id(pattern: WastePattern) -> str:
    """Stable string identifier for serialisation / API responses."""
    match pattern:
        case EmptyRoomHeating():
            return "empty_room_heating_on"
        case OverHeating():
            return "over_heating"


# ---------------------------------------------------------------------------
# Action variants
# ---------------------------------------------------------------------------


@dataclass
class ReduceHeating:
    """Command to lower the heating setpoint on a device."""

    target_device: str


type Action = ReduceHeating


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class EnergyZone(ABC):
    """Every level of the building hierarchy implements this interface."""

    @abstractmethod
    def collect_metrics(self) -> Metrics: ...

    @abstractmethod
    def identify_waste(self) -> list[WastePattern]: ...

    @abstractmethod
    def act(self) -> list[Action]: ...
