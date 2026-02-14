"""FastAPI entry point - thin layer over the domain."""

import asyncio
import dataclasses
import logging
from datetime import datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analysis.adjacency import find_adjacent_rooms
from analysis.thermal import (
    ExternalConditions,
    RoomTimeSeries,
    build_graph_from_adjacency,
    estimate_parameters,
)
from core.models import BuildingLayout, MetricsUpdate
from core.sensors import Sensor
from core.zones import BuildingZone, FloorZone, RoomZone
from data import SAMPLE_BUILDING
from data.electricity_price import ElectricityPriceService
from data.weather import external_temp_at_tick
from services.prediction import PredictionService
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
    global _boost_level

    boost_requested: set[str] = {room_id for room_id, m in update.rooms.items() if "boost_heating" in m.actions}

    for room_id in room_configs:
        room_config = room_configs[room_id]
        is_boosting = room_id in boost_requested
        current_level = _boost_level.get(room_id, 0)

        if is_boosting and current_level == 0:
            _boost_level[room_id] = 1
        elif not is_boosting and current_level > 0:
            _boost_level[room_id] = 0
        elif is_boosting:
            _boost_level[room_id] = min(current_level + 1, _BOOST_RAMP_TICKS)
        elif not is_boosting and current_level > 0:
            _boost_level[room_id] = max(current_level - 1, 0)

        boost_level = _boost_level.get(room_id, 0) / _BOOST_RAMP_TICKS

        room_metrics = update.rooms.get(room_id)
        if room_metrics and "reduce_ventilation" in room_metrics.actions:
            room_config.hvac.max_ventilation_power_w = cfg.reduce_vent_power_w
            room_config.hvac.target_temperature = cfg.control_target_temp_c
            room_config.hvac.heating_gain = cfg.control_heating_gain
            room_config.hvac.max_heating_power_w = cfg.control_max_heating_w
            continue

        if room_metrics and "suspend_heating" in room_metrics.actions:
            room_config.hvac.target_temperature = cfg.suspend_target_temp_c
            room_config.hvac.heating_gain = cfg.suspend_heating_gain
            room_config.hvac.max_heating_power_w = cfg.control_max_heating_w
            continue

        if room_metrics and "reduce_heating" in room_metrics.actions:
            room_config.hvac.target_temperature = cfg.control_target_temp_c
            room_config.hvac.heating_gain = cfg.control_heating_gain
            room_config.hvac.max_heating_power_w = cfg.control_max_heating_w
            continue

        if boost_level > 0:
            room_config.hvac.target_temperature = (
                cfg.control_target_temp_c + (cfg.boost_target_temp_c - cfg.control_target_temp_c) * boost_level
            )
            room_config.hvac.heating_gain = (
                cfg.control_heating_gain + (cfg.boost_heating_gain - cfg.control_heating_gain) * boost_level
            )
            room_config.hvac.max_heating_power_w = (
                cfg.control_max_heating_w + (cfg.boost_max_heating_w - cfg.control_max_heating_w) * boost_level
            )
        else:
            room_config.hvac.target_temperature = cfg.control_target_temp_c
            room_config.hvac.heating_gain = cfg.control_heating_gain
            room_config.hvac.max_heating_power_w = cfg.control_max_heating_w


_forced_power_active: bool = False

# Boost smoothing: track boost ramp level per room (0 = off, 1-_BOOST_RAMP_TICKS = ramping up)
_boost_level: dict[str, int] = {}
_BOOST_RAMP_TICKS: int = 3  # ticks to ramp from normal to boost
_BOOST_RAMP_DOWN_TICKS: int = 5  # ticks to ramp down when boost ends


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
PredictionService.initialize_from_building(SAMPLE_BUILDING)
system_enabled: bool = True

# Thermal estimation history (for collecting data)
_THERMAL_HISTORY_LENGTH = 30  # ticks to collect before estimation
_thermal_by_room: dict[str, list[tuple[datetime, float, float]]] = {}  # room_id -> [(timestamp, temp, power), ...]
_last_temps: dict[str, float] = {}


def _collect_thermal_data(tick: int) -> dict[str, RoomTimeSeries]:
    """Collect historical data for thermal estimation."""
    global _thermal_by_room, _last_temps

    for floor in building.floors:
        for room_zone in floor.rooms:
            room_id = room_zone.room.id
            metrics = room_zone.collect_metrics()
            now = datetime.now()

            if room_id not in _thermal_by_room:
                _thermal_by_room[room_id] = []

            _thermal_by_room[room_id].append((now, metrics.temperature, metrics.heating_power))
            _last_temps[room_id] = metrics.temperature

            # Keep only last N samples
            if len(_thermal_by_room[room_id]) > _THERMAL_HISTORY_LENGTH:
                _thermal_by_room[room_id] = _thermal_by_room[room_id][-_THERMAL_HISTORY_LENGTH:]

    # Build RoomTimeSeries for each room
    room_data: dict[str, RoomTimeSeries] = {}
    for room_id, data in _thermal_by_room.items():
        if len(data) >= 10:
            timestamps = [d[0] for d in data]
            temperatures = [d[1] for d in data]
            powers = [d[2] for d in data]
            room_data[room_id] = RoomTimeSeries(
                room_id=room_id,
                timestamps=timestamps,
                temperature=temperatures,
                heating_power=powers,
            )

    return room_data


def _run_thermal_estimation() -> dict[str, Any] | None:
    """Run thermal parameter estimation and return results."""
    global _thermal_by_room

    if not _thermal_by_room or len(next(iter(_thermal_by_room.values()))) < 10:
        return None

    # Build room data from collected history
    room_data: dict[str, RoomTimeSeries] = {}
    for room_id, data in _thermal_by_room.items():
        if len(data) >= 10:
            timestamps = [d[0] for d in data]
            temperatures = [d[1] for d in data]
            powers = [d[2] for d in data]
            room_data[room_id] = RoomTimeSeries(
                room_id=room_id,
                timestamps=timestamps,
                temperature=temperatures,
                heating_power=powers,
            )
    all_rooms = [room for floor in SAMPLE_BUILDING.floors for room in floor.rooms]
    adjacency: dict[str, list[str]] = {r.id: [] for r in all_rooms}
    for floor in SAMPLE_BUILDING.floors:
        floor_adj = find_adjacent_rooms(floor.rooms)
        for room_id, neighbors in floor_adj.items():
            adjacency[room_id].extend(neighbors)

    # Determine exterior rooms (rooms on building perimeter)
    exterior_rooms: set[str] = set()
    for room in all_rooms:
        poly = room.polygon
        min_x = min(p[0] for p in poly)
        max_x = max(p[0] for p in poly)
        min_y = min(p[1] for p in poly)
        max_y = max(p[1] for p in poly)
        if (
            min_x <= 0.1
            or max_x >= SAMPLE_BUILDING.width_m - 0.1
            or min_y <= 0.1
            or max_y >= SAMPLE_BUILDING.height_m - 0.1
        ):
            exterior_rooms.add(room.id)

    # Build thermal graph
    room_ids = [r.id for r in all_rooms]
    graph = build_graph_from_adjacency(
        room_ids=room_ids,
        adjacency=adjacency,
        exterior_rooms=exterior_rooms,
    )

    # Get room data
    room_data = _collect_thermal_data(0)

    if len(room_data) < 2:
        return None

    # External temperature
    external = ExternalConditions(temperature=external_temp_at_tick(0, 180.0))

    # Run estimation
    result = estimate_parameters(
        room_data=room_data,
        graph=graph,
        external=external,
    )

    return {
        "success": result.success,
        "message": result.message,
        "rmse": result.fit_metrics.rmse,
        "r_squared": result.fit_metrics.r_squared,
        "rooms": {
            room_id: {
                "thermal_mass_j_k": params.thermal_mass,
                "exterior_conductance_w_k": params.exterior_conductance,
            }
            for room_id, params in result.parameters.rooms.items()
        },
        "conductances": {f"{a}-{b}": conductance for (a, b), conductance in result.parameters.conductances.items()},
    }


def _clear_thermal_history() -> None:
    """Clear thermal history."""
    global _thermal_by_room
    _thermal_by_room = {}


class ThermalEstimationResult(BaseModel):
    success: bool
    message: str
    rmse: float
    r_squared: float
    rooms: dict[str, dict[str, float | None]]
    conductances: dict[str, float]


@app.get("/thermal/estimation")
def get_thermal_estimation() -> ThermalEstimationResult | dict[str, str]:
    """Get thermal parameter estimation results."""
    result = _run_thermal_estimation()
    if result is None:
        return {"status": "insufficient_data", "message": "Need more historical data to run estimation"}
    return ThermalEstimationResult(**result)


@app.post("/thermal/estimation/run")
def run_thermal_estimation() -> ThermalEstimationResult | dict[str, str]:
    """Trigger thermal parameter estimation."""
    result = _run_thermal_estimation()
    if result is None:
        return {"status": "insufficient_data", "message": "Need more historical data to run estimation"}
    return ThermalEstimationResult(**result)


@app.post("/thermal/history/clear")
def clear_thermal_history() -> dict[str, str]:
    """Clear thermal estimation history."""
    _clear_thermal_history()
    return {"status": "cleared"}


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


class ElectricityPriceResponse(BaseModel):
    price_nok_per_kwh: float
    valid_from: str
    valid_to: str


@app.get("/electricity/price")
async def get_electricity_price() -> ElectricityPriceResponse:
    """Get current electricity price for Stavanger (NO2 price area)."""
    price = await ElectricityPriceService.get_current_price()
    return ElectricityPriceResponse(
        price_nok_per_kwh=price.price_nok_per_kwh,
        valid_from=price.valid_from.isoformat(),
        valid_to=price.valid_to.isoformat(),
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    tick = 0
    _clear_thermal_history()
    predictions = None

    # Collect initial metrics for prediction before the loop starts
    building.to_metrics_update(tick)

    try:
        while True:
            # Run predictions periodically (every 5 ticks)
            if PredictionService.should_update(tick):
                # Get current state for predictions
                room_metrics = building._collect_room_metrics()
                current_temps = {rid: m.temperature for rid, m in room_metrics.items()}
                heating_powers = {rid: m.heating_power for rid, m in room_metrics.items()}
                neighbor_inflow = {rid: building._net_heat_inflow_w(rid, room_metrics) for rid in room_metrics}
                external_temp = external_temp_at_tick(tick, 180.0)

                # Update thermal estimation if we have enough data
                if len(_thermal_by_room) > 0 and all(len(v) >= 10 for v in _thermal_by_room.values()):
                    room_data: dict[str, RoomTimeSeries] = {}
                    for room_id, samples in _thermal_by_room.items():
                        if len(samples) >= 10:
                            timestamps = [s[0] for s in samples]
                            temps = [s[1] for s in samples]
                            powers = [s[2] for s in samples]
                            room_data[room_id] = RoomTimeSeries(
                                room_id=room_id,
                                timestamps=timestamps,
                                temperature=temps,
                                heating_power=powers,
                            )
                    if room_data:
                        PredictionService.update_estimation(room_data, external_temp)

                # Generate predictions
                predictions = PredictionService.predict(
                    current_temps=current_temps,
                    heating_powers=heating_powers,
                    external_temp=external_temp,
                    neighbor_heat_inflow=neighbor_inflow,
                    tick=tick,
                )
                PredictionService.mark_updated(tick)

            control_snapshot = building.to_metrics_update(tick, predictions)
            if system_enabled:
                _clear_forced_power()
                _apply_actions(control_snapshot)
            else:
                _set_forced_power()
            environment.step(tick)
            update = building.to_metrics_update(tick, predictions)
            update.system_enabled = system_enabled
            update.sun_enabled = environment.sun_enabled

            # Add simulated time (assuming tick_duration_s = 180, starting from a typical day)
            from datetime import datetime, timedelta

            tick_duration_s = 180.0
            base_time = datetime(2025, 1, 15, 6, 0, 0)  # Start at 6 AM
            sim_time = base_time + timedelta(seconds=tick * tick_duration_s)
            update.simulated_time = sim_time.isoformat()

            # Collect thermal data for estimation
            _collect_thermal_data(tick)

            if not system_enabled:
                # Still show waste patterns (for demo value), but clear actions.
                for room_metrics in update.rooms.values():
                    room_metrics.actions = []
            await websocket.send_json(dataclasses.asdict(update))
            tick += 1
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
