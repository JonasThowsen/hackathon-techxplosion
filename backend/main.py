"""FastAPI entry point - thin layer over the domain."""

import asyncio
import dataclasses
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.models import BuildingLayout, MetricsUpdate
from core.sensors import Sensor
from core.zones import BuildingZone, FloorZone, RoomZone
from data import SAMPLE_BUILDING
from simulation import (
    ForceHeatingPower,
    ForceOccupancy,
    SimulatedEnvironment,
    build_room_configs,
    place_sensors,
)

logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")
logging.getLogger("simulation.environment").setLevel(logging.DEBUG)
logging.getLogger("simulation.config_builder").setLevel(logging.INFO)

app = FastAPI(title="FlowMetrics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_zone_hierarchy(
    layout: BuildingLayout,
    sbr: dict[str, list[Sensor]],
    environment: SimulatedEnvironment,
) -> BuildingZone:
    floor_zones: list[FloorZone] = []
    for floor in layout.floors:
        room_zones = [RoomZone(room, sbr[room.id], environment) for room in floor.rooms]
        floor_zones.append(FloorZone(floor, room_zones))
    return BuildingZone(layout, floor_zones)


def _apply_actions(update: MetricsUpdate) -> None:
    """Apply control actions back into the simulation config."""
    cfg = environment.config
    for room_config in room_configs.values():
        room_config.hvac.target_temperature = cfg.control_target_temp_c
        room_config.hvac.heating_gain = cfg.control_heating_gain
        room_config.hvac.max_heating_power_w = cfg.control_max_heating_w

    for room_id, room_metrics in update.rooms.items():
        room_config = room_configs.get(room_id)
        if room_config is None:
            continue

        if "boost_heating" in room_metrics.actions:
            room_config.hvac.target_temperature = cfg.boost_target_temp_c
            room_config.hvac.heating_gain = cfg.boost_heating_gain
            room_config.hvac.max_heating_power_w = cfg.boost_max_heating_w
            continue

        if "reduce_ventilation" in room_metrics.actions:
            room_config.hvac.max_ventilation_power_w = cfg.reduce_vent_power_w

        if "suspend_heating" in room_metrics.actions:
            room_config.hvac.target_temperature = cfg.suspend_target_temp_c
            room_config.hvac.heating_gain = cfg.suspend_heating_gain

        if "reduce_heating" in room_metrics.actions:
            room_config.hvac.target_temperature = cfg.control_target_temp_c


_forced_power_active: bool = False


def _set_forced_power() -> None:
    """Add ForceHeatingPower for all rooms (system OFF → constant burn)."""
    global _forced_power_active
    if not _forced_power_active:
        for room_id in room_configs:
            environment.add_scenario(room_id, ForceHeatingPower(power_w=environment.config.forced_power_w))
        _forced_power_active = True


def _clear_forced_power() -> None:
    """Remove all ForceHeatingPower scenarios, preserving other scenarios."""
    global _forced_power_active
    if _forced_power_active:
        environment.scenarios = [s for s in environment.scenarios if not isinstance(s.scenario, ForceHeatingPower)]
        _forced_power_active = False


def _configure_demo_scenarios() -> None:
    """Pin occupancy for demo rooms so empty-room patterns are stable."""
    environment.add_scenario("r-003", ForceOccupancy(occupied=False))
    environment.add_scenario("r-105", ForceOccupancy(occupied=False))


# --- module-level state, initialised at import time ---
sensors_by_room: dict[str, list[Sensor]] = place_sensors(SAMPLE_BUILDING)
room_configs = build_room_configs(SAMPLE_BUILDING)
_room_names = {room.id: room.name for floor in SAMPLE_BUILDING.floors for room in floor.rooms}
environment = SimulatedEnvironment(room_configs, room_names=_room_names)
_configure_demo_scenarios()
building: BuildingZone = _build_zone_hierarchy(SAMPLE_BUILDING, sensors_by_room, environment)
system_enabled: bool = True


class ToggleStatus(BaseModel):
    enabled: bool


@app.get("/building")
def get_building() -> BuildingLayout:
    return SAMPLE_BUILDING


@app.get("/system/status")
def get_system_status() -> ToggleStatus:
    return ToggleStatus(enabled=system_enabled)


@app.post("/system/toggle")
def toggle_system() -> ToggleStatus:
    global system_enabled
    system_enabled = not system_enabled
    return ToggleStatus(enabled=system_enabled)


@app.get("/sun/status")
def get_sun_status() -> ToggleStatus:
    return ToggleStatus(enabled=environment.sun_enabled)


@app.post("/sun/toggle")
def toggle_sun() -> ToggleStatus:
    environment.sun_enabled = not environment.sun_enabled
    return ToggleStatus(enabled=environment.sun_enabled)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    tick = 0
    try:
        while True:
            control_snapshot = building.to_metrics_update(tick)
            if system_enabled:
                _clear_forced_power()
                _apply_actions(control_snapshot)
            else:
                _set_forced_power()
            environment.step(tick)
            update = building.to_metrics_update(tick)
            update.system_enabled = system_enabled
            update.sun_enabled = environment.sun_enabled
            if not system_enabled:
                # Still show waste patterns (for demo value), but clear actions.
                for room_metrics in update.rooms.values():
                    room_metrics.actions = []
            await websocket.send_json(dataclasses.asdict(update))
            tick += 1
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
