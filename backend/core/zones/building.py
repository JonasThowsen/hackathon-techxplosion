"""BuildingZone - top-level zone aggregating all floors."""

from typing import override

from analysis import calculate_heat_flows, find_adjacent_rooms, net_heat_flow_by_room
from core.models import BuildingLayout, MetricsUpdate, Room, RoomMetrics
from core.zones.base import Action, EnergyZone, Metrics, WastePattern, waste_pattern_id
from core.zones.floor import FloorZone


class BuildingZone(EnergyZone):
    """Top-level zone aggregating all floors in a building."""

    def __init__(self, layout: BuildingLayout, floors: list[FloorZone]) -> None:
        self.layout = layout
        self.floors = floors

    @override
    def collect_metrics(self) -> Metrics:
        if not self.floors:
            return Metrics(temperature=0.0, occupancy=False, co2=0.0, power=0.0)

        floor_metrics = [f.collect_metrics() for f in self.floors]
        n = len(floor_metrics)
        return Metrics(
            temperature=sum(m.temperature for m in floor_metrics) / n,
            occupancy=any(m.occupancy for m in floor_metrics),
            co2=sum(m.co2 for m in floor_metrics) / n,
            power=sum(m.power for m in floor_metrics),
        )

    @override
    def identify_waste(self) -> list[WastePattern]:
        patterns: list[WastePattern] = []
        for floor in self.floors:
            patterns.extend(floor.identify_waste())
        return patterns

    @override
    def act(self) -> list[Action]:
        actions: list[Action] = []
        for floor in self.floors:
            actions.extend(floor.act())
        return actions

    def to_metrics_update(self, tick: int) -> MetricsUpdate:
        """Generate a MetricsUpdate for the WebSocket stream."""
        room_metrics_map: dict[str, Metrics] = {}
        room_waste_map: dict[str, list[WastePattern]] = {}
        rooms: dict[str, RoomMetrics] = {}
        all_rooms: list[Room] = []

        for floor_zone in self.floors:
            for room_zone in floor_zone.rooms:
                metrics = room_zone.collect_metrics()
                waste = room_zone.identify_waste()
                room_metrics_map[room_zone.room.id] = metrics
                room_waste_map[room_zone.room.id] = waste
                all_rooms.append(room_zone.room)
                rooms[room_zone.room.id] = RoomMetrics(
                    temperature=metrics.temperature,
                    occupancy=metrics.occupancy,
                    co2=metrics.co2,
                    power=metrics.power,
                    waste_patterns=[waste_pattern_id(p) for p in waste],
                )

        # Compute heat flows and apply net flow to each room
        adjacency = find_adjacent_rooms(all_rooms)
        flows = calculate_heat_flows(room_metrics_map, adjacency, all_rooms)
        net_flows = net_heat_flow_by_room(flows)

        for room_id, net in net_flows.items():
            if room_id in rooms:
                rooms[room_id].heat_flow = round(net, 1)

        return MetricsUpdate(tick=tick, rooms=rooms)
