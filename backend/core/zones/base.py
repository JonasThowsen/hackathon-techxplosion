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


@dataclass
class OpenWindowHeating:
    """Heating running while temperature is dropping (likely open window)."""

    room_name: str
    estimated_kwh_wasted: float
    duration_minutes: float
    temp_drop_rate: float  # °C per tick


@dataclass
class ExcessiveVentilation:
    """Ventilation running at full in unoccupied room with good air quality."""

    room_name: str
    estimated_kwh_wasted: float
    duration_minutes: float


type WastePattern = EmptyRoomHeating | OverHeating | OpenWindowHeating | ExcessiveVentilation


def waste_pattern_id(pattern: WastePattern) -> str:
    """Stable string identifier for serialisation / API responses."""
    match pattern:
        case EmptyRoomHeating():
            return "empty_room_heating_on"
        case OverHeating():
            return "over_heating"
        case OpenWindowHeating():
            return "open_window_heating"
        case ExcessiveVentilation():
            return "excessive_ventilation"


# ---------------------------------------------------------------------------
# Action variants
# ---------------------------------------------------------------------------


@dataclass
class ReduceHeating:
    """Command to lower the heating setpoint on a device."""

    target_device: str


@dataclass
class BoostHeating:
    """Command to temporarily raise heating to recover baseline comfort."""

    target_device: str


@dataclass
class ReduceVentilation:
    """Command to lower ventilation in an unoccupied room."""

    target_device: str


@dataclass
class OpenWindowAlert:
    """Alert that a window appears to be open - reduce heating."""

    target_device: str


type Action = ReduceHeating | BoostHeating | ReduceVentilation | OpenWindowAlert


def action_id(action: Action) -> str:
    """Stable string identifier for serialisation / API responses."""
    match action:
        case ReduceHeating():
            return "reduce_heating"
        case BoostHeating():
            return "boost_heating"
        case ReduceVentilation():
            return "reduce_ventilation"
        case OpenWindowAlert():
            return "open_window_alert"


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
