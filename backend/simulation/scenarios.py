"""Simulation scenarios - controllable events and faults."""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Fault scenarios - equipment malfunctions
# ---------------------------------------------------------------------------


@dataclass
class ThermostatStuck:
    """Thermostat stuck at a fixed reading, heating doesn't respond correctly."""

    stuck_reading: float  # What the thermostat "sees"


@dataclass
class HeatingStuckOn:
    """Heating valve stuck open - always heating at fixed power."""

    power_w: float


@dataclass
class HeatingStuckOff:
    """Heating valve stuck closed - no heating available."""

    pass


@dataclass
class VentilationFailed:
    """Ventilation system not working - no forced air exchange."""

    pass


type FaultScenario = ThermostatStuck | HeatingStuckOn | HeatingStuckOff | VentilationFailed


# ---------------------------------------------------------------------------
# Event scenarios - temporary conditions
# ---------------------------------------------------------------------------


@dataclass
class WindowOpen:
    """Window left open - increased heat loss to exterior."""

    # Equivalent U-value increase (W/K) - how much extra heat loss
    heat_loss_w_per_k: float = 50.0


@dataclass
class DoorPropped:
    """Door propped open to another room - increased air exchange."""

    neighbor_room_id: str
    # Equivalent U-value for the opening (W/K)
    heat_exchange_w_per_k: float = 100.0


@dataclass
class ExternalTempOverride:
    """Override external temperature for this room's exterior walls."""

    temperature: float


type EventScenario = WindowOpen | DoorPropped | ExternalTempOverride


# ---------------------------------------------------------------------------
# Override scenarios - force specific states
# ---------------------------------------------------------------------------


@dataclass
class ForceOccupancy:
    """Force room occupancy state."""

    occupied: bool


@dataclass
class ForceHeatingPower:
    """Force heating to specific power level."""

    power_w: float


@dataclass
class ForceTemperature:
    """Force room temperature (useful for testing)."""

    temperature: float


type OverrideScenario = ForceOccupancy | ForceHeatingPower | ForceTemperature


# ---------------------------------------------------------------------------
# Combined scenario type
# ---------------------------------------------------------------------------

type Scenario = FaultScenario | EventScenario | OverrideScenario


@dataclass
class ActiveScenario:
    """A scenario applied to a room with optional time bounds."""

    room_id: str
    scenario: Scenario
    start_tick: int | None = None  # None = always active
    end_tick: int | None = None  # None = no end


def is_scenario_active(active: ActiveScenario, tick: int) -> bool:
    """Check if a scenario is active at the given tick."""
    if active.start_tick is not None and tick < active.start_tick:
        return False
    return not (active.end_tick is not None and tick >= active.end_tick)
