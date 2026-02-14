"""Action command variants issued by the zone hierarchy."""

from dataclasses import dataclass


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
class SuspendHeating:
    """Suspend heating due to rapid heat loss."""

    target_device: str


type Action = ReduceHeating | BoostHeating | ReduceVentilation | SuspendHeating


def action_id(action: Action) -> str:
    """Stable string identifier for serialisation / API responses."""
    match action:
        case ReduceHeating():
            return "reduce_heating"
        case BoostHeating():
            return "boost_heating"
        case ReduceVentilation():
            return "reduce_ventilation"
        case SuspendHeating():
            return "suspend_heating"
