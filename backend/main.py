"""FastAPI entry point - thin layer over the domain."""

import asyncio
import dataclasses

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from core.models import BuildingLayout
from core.sensors import Sensor
from core.zones import BuildingZone, FloorZone, RoomZone
from data import SAMPLE_BUILDING
from simulation import place_sensors, simulate_tick

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
) -> BuildingZone:
    floor_zones: list[FloorZone] = []
    for floor in layout.floors:
        room_zones = [RoomZone(room, sbr[room.id]) for room in floor.rooms]
        floor_zones.append(FloorZone(floor, room_zones))
    return BuildingZone(layout, floor_zones)


# --- module-level state, initialised at import time ---
sensors_by_room: dict[str, list[Sensor]] = place_sensors(SAMPLE_BUILDING)
building: BuildingZone = _build_zone_hierarchy(SAMPLE_BUILDING, sensors_by_room)


@app.get("/building")
def get_building() -> BuildingLayout:
    return SAMPLE_BUILDING


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    tick = 0
    try:
        while True:
            simulate_tick(sensors_by_room, tick)
            update = building.to_metrics_update(tick)
            await websocket.send_json(dataclasses.asdict(update))
            tick += 1
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass
