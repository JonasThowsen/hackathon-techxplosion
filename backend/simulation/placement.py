"""Sensor placement logic."""

from core.models import BuildingLayout
from core.sensors import Sensor, SensorKind


def place_sensors(building: BuildingLayout) -> dict[str, list[Sensor]]:
    """Place sensors in each room. Returns room_id -> list[Sensor].

    For each room, create one sensor of each kind (temp, occupancy, co2, power, light).
    Position each sensor at the centroid of the room polygon.
    """
    result: dict[str, list[Sensor]] = {}
    for floor in building.floors:
        for room in floor.rooms:
            cx = sum(p[0] for p in room.polygon) / len(room.polygon)
            cy = sum(p[1] for p in room.polygon) / len(room.polygon)
            sensors: list[Sensor] = []
            for kind in SensorKind:
                sensor = Sensor(
                    id=f"{room.id}-{kind.value}",
                    kind=kind,
                    x=cx,
                    y=cy,
                    floor=floor.floor_index,
                )
                sensors.append(sensor)
            result[room.id] = sensors
    return result
