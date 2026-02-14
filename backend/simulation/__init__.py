"""Simulation module - room physics simulation and sensor placement."""

from simulation.config_builder import build_room_configs
from simulation.environment import RoomEnvironmentSource, SimulatedEnvironment
from simulation.placement import place_sensors
from simulation.room_config import HVACConfig, RoomPhysicsConfig, WallConfig
from simulation.room_state import RoomState
from simulation.scenarios import (
    ActiveScenario,
    DoorPropped,
    ExternalTempOverride,
    ForceHeatingPower,
    ForceOccupancy,
    ForceTemperature,
    HeatingStuckOff,
    HeatingStuckOn,
    Scenario,
    ThermostatStuck,
    VentilationFailed,
    WindowOpen,
)

__all__ = [
    "ActiveScenario",
    "DoorPropped",
    "ExternalTempOverride",
    "ForceHeatingPower",
    "ForceOccupancy",
    "ForceTemperature",
    "HVACConfig",
    "HeatingStuckOff",
    "HeatingStuckOn",
    "RoomEnvironmentSource",
    "RoomPhysicsConfig",
    "RoomState",
    "Scenario",
    "SimulatedEnvironment",
    "ThermostatStuck",
    "VentilationFailed",
    "WallConfig",
    "WindowOpen",
    "build_room_configs",
    "place_sensors",
]
