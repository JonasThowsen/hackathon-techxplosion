from energy_zone import Action, EnergyZone, Metrics, WastePattern
from models import Floor
from room_zone import RoomZone


class FloorZone(EnergyZone):
    """Aggregates RoomZones on the same floor."""

    def __init__(self, floor: Floor, rooms: list[RoomZone]) -> None:
        self.floor = floor
        self.rooms = rooms

    def collect_metrics(self) -> Metrics:
        if not self.rooms:
            return Metrics(temperature=0.0, occupancy=False, co2=0.0, power=0.0)

        room_metrics = [r.collect_metrics() for r in self.rooms]
        n = len(room_metrics)
        return Metrics(
            temperature=sum(m.temperature for m in room_metrics) / n,
            occupancy=any(m.occupancy for m in room_metrics),
            co2=sum(m.co2 for m in room_metrics) / n,
            power=sum(m.power for m in room_metrics),
        )

    def identify_waste(self) -> list[WastePattern]:
        patterns: list[WastePattern] = []
        for room in self.rooms:
            patterns.extend(room.identify_waste())
        return patterns

    def act(self) -> list[Action]:
        actions: list[Action] = []
        for room in self.rooms:
            actions.extend(room.act())
        return actions
