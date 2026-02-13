from energy_zone import Action, EnergyZone, Metrics, WastePattern
from floor_zone import FloorZone
from models import BuildingLayout, MetricsUpdate, RoomMetrics


class BuildingZone(EnergyZone):
    """Top-level zone aggregating all floors in a building."""

    def __init__(self, layout: BuildingLayout, floors: list[FloorZone]) -> None:
        self.layout = layout
        self.floors = floors

    def collect_metrics(self) -> Metrics:
        if not self.floors:
            return Metrics(temperature=0.0, occupancy=0.0, co2=0.0, power=0.0)

        floor_metrics = [f.collect_metrics() for f in self.floors]
        n = len(floor_metrics)
        return Metrics(
            temperature=sum(m.temperature for m in floor_metrics) / n,
            occupancy=sum(m.occupancy for m in floor_metrics) / n,
            co2=sum(m.co2 for m in floor_metrics) / n,
            power=sum(m.power for m in floor_metrics),
        )

    def identify_waste(self) -> list[WastePattern]:
        patterns: list[WastePattern] = []
        for floor in self.floors:
            patterns.extend(floor.identify_waste())
        return patterns

    def act(self) -> list[Action]:
        actions: list[Action] = []
        for floor in self.floors:
            actions.extend(floor.act())
        return actions

    def to_metrics_update(self, tick: int) -> MetricsUpdate:
        rooms: dict[str, RoomMetrics] = {}
        for floor_zone in self.floors:
            for room_zone in floor_zone.rooms:
                metrics = room_zone.collect_metrics()
                waste = room_zone.identify_waste()
                rooms[room_zone.room.id] = RoomMetrics(
                    temperature=metrics.temperature,
                    occupancy=metrics.occupancy >= 0.5,
                    co2=metrics.co2,
                    power=metrics.power,
                    waste_patterns=[p.pattern_id for p in waste],
                )
        return MetricsUpdate(tick=tick, rooms=rooms)
