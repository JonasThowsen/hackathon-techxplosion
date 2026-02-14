"""BuildingZone - top-level zone aggregating all floors."""

from collections import defaultdict
from typing import override

from analysis.adjacency import find_adjacent_rooms, find_vertical_neighbours
from core.models import BuildingLayout, HeatFlow, MetricsUpdate, RoomMetrics
from core.zones.base import (
    Action,
    BoostHeating,
    EmptyRoomHeating,
    EnergyZone,
    ExcessiveVentilation,
    Metrics,
    OpenWindowAlert,
    OpenWindowHeating,
    OverHeating,
    ReduceHeating,
    ReduceVentilation,
    WastePattern,
    action_id,
    waste_pattern_id,
)
from core.zones.floor import FloorZone

_BASELINE_TEMPERATURE_C = 21.0
_GOOD_AIR_QUALITY_CO2_PPM = 900.0
_HEATING_ACTIVE_W = 120.0
_OVERHEATING_MARGIN_C = 1.0
_NEIGHBOUR_DEVIATION_C = 0.7
_TICK_DURATION_MINUTES = 3.0
_COLD_MARGIN_C = 0.2
_OPEN_WINDOW_DROP_THRESHOLD_C = 0.3  # min temp drop over 3 ticks to flag
_EXCESSIVE_VENT_CO2_PPM = 500.0
_EXCESSIVE_VENT_POWER_W = 50.0
_TEMP_HISTORY_LENGTH = 5
_SIGNIFICANT_INFLOW_W = 200.0  # skip boost if receiving this much free heat
_SIGNIFICANT_OUTFLOW_W = 200.0  # reduce heating if losing this much to neighbours


class BuildingZone(EnergyZone):
    """Top-level zone aggregating all floors in a building."""

    def __init__(self, layout: BuildingLayout, floors: list[FloorZone]) -> None:
        self.layout = layout
        self.floors = floors
        self._room_name_to_id = {room.name: room.id for floor in self.layout.floors for room in floor.rooms}
        self._last_room_metrics: dict[str, Metrics] = {}
        self._neighbours = self.calculate_neighbours()
        self._temp_history: dict[str, list[float]] = {}

    def calculate_neighbours(self) -> dict[str, list[str]]:
        """Calculate room neighbours: same-floor walls + floor/ceiling overlaps."""
        neighbours: dict[str, list[str]] = {}
        # Horizontal: shared walls on the same floor
        for floor in self.layout.floors:
            floor_adj = find_adjacent_rooms(floor.rooms)
            neighbours.update(floor_adj)
        # Vertical: rooms directly above/below on adjacent floors
        vertical = find_vertical_neighbours(self.layout.floors)
        for room_id, v_neighbours in vertical.items():
            neighbours.setdefault(room_id, []).extend(v_neighbours)
        return neighbours

    def _collect_room_metrics(self) -> dict[str, Metrics]:
        room_metrics: dict[str, Metrics] = {}
        for floor_zone in self.floors:
            for room_zone in floor_zone.rooms:
                room_metrics[room_zone.room.id] = room_zone.collect_metrics()
        self._last_room_metrics = room_metrics

        # Track temperature history for rate-of-change analysis.
        for room_id, metrics in room_metrics.items():
            history = self._temp_history.setdefault(room_id, [])
            history.append(metrics.temperature)
            if len(history) > _TEMP_HISTORY_LENGTH:
                history.pop(0)

        return room_metrics

    def _room_id_from_pattern(self, pattern: WastePattern) -> str:
        pattern_room = pattern.room_name
        return self._room_name_to_id.get(pattern_room, pattern_room)

    def _temp_drop_rate(self, room_id: str) -> float:
        """Return average temperature drop per tick over the last 3 readings."""
        history = self._temp_history.get(room_id, [])
        if len(history) < 3:
            return 0.0
        recent = history[-3:]
        return (recent[0] - recent[-1]) / (len(recent) - 1)

    def _identify_waste_from_metrics(self, room_metrics: dict[str, Metrics]) -> list[WastePattern]:
        patterns: list[WastePattern] = []

        for room_id, metrics in room_metrics.items():
            neighbour_ids = self._neighbours.get(room_id, [])
            neighbour_temps = [room_metrics[n].temperature for n in neighbour_ids if n in room_metrics]
            neighbour_avg = sum(neighbour_temps) / len(neighbour_temps) if neighbour_temps else _BASELINE_TEMPERATURE_C

            hotter_than_neighbours = metrics.temperature > neighbour_avg + _NEIGHBOUR_DEVIATION_C

            # Open window detection: temp dropping while heating is active.
            drop_rate = self._temp_drop_rate(room_id)
            if drop_rate > _OPEN_WINDOW_DROP_THRESHOLD_C and metrics.heating_power >= _HEATING_ACTIVE_W:
                patterns.append(
                    OpenWindowHeating(
                        room_name=room_id,
                        estimated_kwh_wasted=round(
                            (metrics.heating_power / 1000) * (_TICK_DURATION_MINUTES / 60),
                            3,
                        ),
                        duration_minutes=_TICK_DURATION_MINUTES,
                        temp_drop_rate=round(drop_rate, 3),
                    )
                )

            # Excessive ventilation: unoccupied, good air quality, ventilation still running high.
            if (
                not metrics.occupancy
                and metrics.co2 < _EXCESSIVE_VENT_CO2_PPM
                and metrics.ventilation_power > _EXCESSIVE_VENT_POWER_W
            ):
                patterns.append(
                    ExcessiveVentilation(
                        room_name=room_id,
                        estimated_kwh_wasted=round(
                            (metrics.ventilation_power / 1000) * (_TICK_DURATION_MINUTES / 60),
                            3,
                        ),
                        duration_minutes=_TICK_DURATION_MINUTES,
                    )
                )

            # Empty room heating.
            if (
                not metrics.occupancy
                and metrics.heating_power >= _HEATING_ACTIVE_W
                and (metrics.temperature > _BASELINE_TEMPERATURE_C + 0.5 or hotter_than_neighbours)
            ):
                patterns.append(
                    EmptyRoomHeating(
                        room_name=room_id,
                        estimated_kwh_wasted=round(
                            (metrics.heating_power / 1000) * (_TICK_DURATION_MINUTES / 60),
                            3,
                        ),
                        duration_minutes=_TICK_DURATION_MINUTES,
                    )
                )

            # Overheating: only flag if neighbours are not equally warm (avoids false positives from solar gain).
            if (
                metrics.temperature > _BASELINE_TEMPERATURE_C + _OVERHEATING_MARGIN_C
                and metrics.heating_power > 0
                and hotter_than_neighbours
            ):
                patterns.append(
                    OverHeating(
                        room_name=room_id,
                        estimated_kwh_wasted=round(
                            (metrics.heating_power / 1000) * (_TICK_DURATION_MINUTES / 60),
                            3,
                        ),
                        duration_minutes=_TICK_DURATION_MINUTES,
                    )
                )

        return patterns

    def _net_heat_inflow_w(self, room_id: str, room_metrics: dict[str, Metrics]) -> float:
        """Estimate net heat flowing INTO a room from all neighbours (watts).

        Positive means neighbours are warming this room; negative means this
        room is losing heat to neighbours.  Used to make smarter control
        decisions -- e.g. skip boost-heating when free heat is already arriving.
        """
        metrics = room_metrics.get(room_id)
        if metrics is None:
            return 0.0
        total = 0.0
        for neighbour_id in self._neighbours.get(room_id, []):
            neighbour = room_metrics.get(neighbour_id)
            if neighbour is None:
                continue
            # Positive dt means neighbour is warmer → heat flows in
            dt = neighbour.temperature - metrics.temperature
            # Rough conductance estimate (same factor used in to_metrics_update)
            total += dt * 1.5 * 27
        return total

    def _actions_from_metrics(
        self,
        room_metrics: dict[str, Metrics],
        waste_patterns: list[WastePattern],
    ) -> list[Action]:
        actions: list[Action] = []
        acted_rooms: set[str] = set()

        # Comfort recovery: if rooms drift too far below baseline, boost heating.
        # But skip the boost if the room is already receiving significant free
        # heat from neighbours (heat bleed working in our favour).
        for room_id, metrics in room_metrics.items():
            if (
                metrics.occupancy
                and metrics.temperature <= _BASELINE_TEMPERATURE_C - _COLD_MARGIN_C
                and metrics.co2 <= _GOOD_AIR_QUALITY_CO2_PPM
            ):
                inflow = self._net_heat_inflow_w(room_id, room_metrics)
                if inflow < _SIGNIFICANT_INFLOW_W:
                    actions.append(BoostHeating(target_device=room_id))
                    acted_rooms.add(room_id)

        # Generate targeted actions from waste patterns.
        for pattern in waste_patterns:
            room_id = self._room_id_from_pattern(pattern)
            if room_id in acted_rooms:
                continue

            metrics = room_metrics.get(room_id)
            if metrics is None:
                continue

            match pattern:
                case OpenWindowHeating():
                    actions.append(OpenWindowAlert(target_device=room_id))
                    acted_rooms.add(room_id)
                case ExcessiveVentilation():
                    actions.append(ReduceVentilation(target_device=room_id))
                    acted_rooms.add(room_id)
                case EmptyRoomHeating():
                    if metrics.co2 <= _GOOD_AIR_QUALITY_CO2_PPM:
                        actions.append(ReduceHeating(target_device=room_id))
                        acted_rooms.add(room_id)
                case OverHeating():
                    if metrics.co2 <= _GOOD_AIR_QUALITY_CO2_PPM:
                        actions.append(ReduceHeating(target_device=room_id))
                        acted_rooms.add(room_id)

        # Heat-flow-aware reduction: if a room is actively heating but losing
        # significant heat to cold neighbours (and is unoccupied), reduce its
        # heating to stop wasting energy warming other spaces.
        for room_id, metrics in room_metrics.items():
            if room_id in acted_rooms:
                continue
            if metrics.heating_power < _HEATING_ACTIVE_W:
                continue
            if metrics.occupancy:
                continue
            outflow = -self._net_heat_inflow_w(room_id, room_metrics)
            if outflow > _SIGNIFICANT_OUTFLOW_W and metrics.co2 <= _GOOD_AIR_QUALITY_CO2_PPM:
                actions.append(ReduceHeating(target_device=room_id))
                acted_rooms.add(room_id)

        return actions

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
        room_metrics = self._collect_room_metrics()
        return self._identify_waste_from_metrics(room_metrics)

    @override
    def act(self) -> list[Action]:
        room_metrics = self._last_room_metrics if self._last_room_metrics else self._collect_room_metrics()
        waste_patterns = self._identify_waste_from_metrics(room_metrics)
        return self._actions_from_metrics(room_metrics, waste_patterns)

    def to_metrics_update(self, tick: int) -> MetricsUpdate:
        """Generate a MetricsUpdate for the WebSocket stream."""
        room_metrics = self._collect_room_metrics()
        waste_patterns = self._identify_waste_from_metrics(room_metrics)
        actions = self._actions_from_metrics(room_metrics, waste_patterns)

        patterns_by_room: dict[str, list[str]] = defaultdict(list)
        for pattern in waste_patterns:
            room_id = self._room_id_from_pattern(pattern)
            patterns_by_room[room_id].append(waste_pattern_id(pattern))

        actions_by_room: dict[str, list[str]] = defaultdict(list)
        for action in actions:
            match action:
                case BoostHeating(target_device=device):
                    actions_by_room[device].append(action_id(action))
                case ReduceHeating(target_device=device):
                    actions_by_room[device].append(action_id(action))
                case ReduceVentilation(target_device=device):
                    actions_by_room[device].append(action_id(action))
                case OpenWindowAlert(target_device=device):
                    actions_by_room[device].append(action_id(action))

        rooms: dict[str, RoomMetrics] = {}

        for room_id, metrics in room_metrics.items():
            rooms[room_id] = RoomMetrics(
                temperature=metrics.temperature,
                occupancy=metrics.occupancy,
                co2=metrics.co2,
                heating_power=metrics.heating_power,
                ventilation_power=metrics.ventilation_power,
                waste_patterns=patterns_by_room.get(room_id, []),
                actions=actions_by_room.get(room_id, []),
            )

        heat_flows: list[HeatFlow] = []
        for room_id, metrics in room_metrics.items():
            for neighbour_id in self._neighbours.get(room_id, []):
                neighbour = room_metrics.get(neighbour_id)
                if neighbour is None:
                    continue
                dt = metrics.temperature - neighbour.temperature
                if dt > 0.5:  # only significant flows
                    heat_flows.append(HeatFlow(from_room=room_id, to_room=neighbour_id, watts=round(dt * 1.5 * 27, 1)))

        return MetricsUpdate(tick=tick, rooms=rooms, heat_flows=heat_flows)
