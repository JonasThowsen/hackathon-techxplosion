"""Simulation module - room physics simulation and sensor placement."""

from simulation.config import DEFAULT as DEFAULT_SIM_CONFIG
from simulation.config import SimConfig
from simulation.config_builder import build_room_configs
from simulation.environment import RoomEnvironmentSource, SimulatedEnvironment
from simulation.placement import place_sensors
from simulation.room_config import HVACConfig, RoomPhysicsConfig, WallConfig, WindowConfig
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
    "DEFAULT_SIM_CONFIG",
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
    "SimConfig",
    "SimulatedEnvironment",
    "ThermostatStuck",
    "VentilationFailed",
    "WallConfig",
    "WindowConfig",
    "WindowOpen",
    "build_room_configs",
    "place_sensors",
]
