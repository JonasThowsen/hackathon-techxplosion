"""BuildingZone - top-level zone aggregating all floors."""

from typing import override

from core.models import BuildingLayout, MetricsUpdate, RoomMetrics
from core.zones.base import Action, EnergyZone, Metrics, WastePattern
from core.zones.floor import FloorZone


class BuildingZone(EnergyZone):
    """Top-level zone aggregating all floors in a building."""

    def __init__(self, layout: BuildingLayout, floors: list[FloorZone]) -> None:
        self.layout = layout
        self.floors = floors

    @override
    def collect_metrics(self) -> Metrics:
        if not self.floors:
            return Metrics(
                temperature=0.0,
                occupancy=False,
                co2=0.0,
                heating_power=0.0,
                ventilation_power=0.0,
            )

        floor_metrics = [f.collect_metrics() for f in self.floors]
        n = len(floor_metrics)
        return Metrics(
            temperature=sum(m.temperature for m in floor_metrics) / n,
            occupancy=any(m.occupancy for m in floor_metrics),
            co2=sum(m.co2 for m in floor_metrics) / n,
            heating_power=sum(m.heating_power for m in floor_metrics),
            ventilation_power=sum(m.ventilation_power for m in floor_metrics),
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
        rooms: dict[str, RoomMetrics] = {}

        for floor_zone in self.floors:
            for room_zone in floor_zone.rooms:
                metrics = room_zone.collect_metrics()
                rooms[room_zone.room.id] = RoomMetrics(
                    temperature=metrics.temperature,
                    occupancy=metrics.occupancy,
                    co2=metrics.co2,
                    heating_power=metrics.heating_power,
                    ventilation_power=metrics.ventilation_power,
                )

        return MetricsUpdate(tick=tick, rooms=rooms)
