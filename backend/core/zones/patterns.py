"""Waste pattern variants detected by the zone hierarchy."""

from dataclasses import dataclass


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
class RapidHeatLoss:
    """Heating running while temperature drops rapidly (unexplained heat loss)."""

    room_name: str
    estimated_kwh_wasted: float
    duration_minutes: float
    heat_loss_rate: float  # °C per tick


@dataclass
class ExcessiveVentilation:
    """Ventilation running at full in unoccupied room with good air quality."""

    room_name: str
    estimated_kwh_wasted: float
    duration_minutes: float


type WastePattern = EmptyRoomHeating | OverHeating | RapidHeatLoss | ExcessiveVentilation


def waste_pattern_id(pattern: WastePattern) -> str:
    """Stable string identifier for serialisation / API responses."""
    match pattern:
        case EmptyRoomHeating():
            return "empty_room_heating_on"
        case OverHeating():
            return "over_heating"
        case RapidHeatLoss():
            return "rapid_heat_loss"
        case ExcessiveVentilation():
            return "excessive_ventilation"
