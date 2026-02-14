"""RoomZone - atomic energy zone wrapping a single room."""

from typing import TYPE_CHECKING, override

from core.models import Room
from core.sensors import Sensor, SensorKind
from core.zones.base import (
    Action,
    EmptyRoomHeating,
    EnergyZone,
    Metrics,
    OverHeating,
    ReduceHeating,
    WastePattern,
)

if TYPE_CHECKING:
    from simulation.environment import RoomEnvironmentSource


class RoomZone(EnergyZone):
    """Atomic energy zone wrapping a single room and its sensors."""

    def __init__(
        self,
        room: Room,
        sensors: list[Sensor],
        environment: RoomEnvironmentSource,
    ) -> None:
        self.room = room
        self.sensors = sensors
        self.environment = environment
        self._last_metrics: Metrics | None = None

    def _readings_by_kind(self, kind: SensorKind) -> list[float]:
        return [s.read(self.environment) for s in self.sensors if s.kind == kind]

    @override
    def collect_metrics(self) -> Metrics:
        temps = self._readings_by_kind(SensorKind.TEMPERATURE)
        occ = self._readings_by_kind(SensorKind.OCCUPANCY)
        co2 = self._readings_by_kind(SensorKind.CO2)
        heating = self._readings_by_kind(SensorKind.HEATING_POWER)
        ventilation = self._readings_by_kind(SensorKind.VENTILATION_POWER)

        metrics = Metrics(
            temperature=sum(temps) / len(temps) if temps else 0.0,
            occupancy=any(v > 0.5 for v in occ),
            co2=sum(co2) / len(co2) if co2 else 0.0,
            heating_power=sum(heating),
            ventilation_power=sum(ventilation),
        )
        self._last_metrics = metrics
        return metrics

    @override
    def identify_waste(self) -> list[WastePattern]:
        m = self._last_metrics
        if m is None:
            return []

        patterns: list[WastePattern] = []

        # Heating running in empty room
        if not m.occupancy and m.heating_power > 100.0:
            patterns.append(
                EmptyRoomHeating(
                    room_name=self.room.name,
                    estimated_kwh_wasted=m.heating_power * 0.001,
                    duration_minutes=5.0,
                )
            )

        # Over-heating (above comfort threshold)
        if m.temperature > 24.0:
            patterns.append(
                OverHeating(
                    room_name=self.room.name,
                    estimated_kwh_wasted=(m.temperature - 22.0) * 0.05,
                    duration_minutes=5.0,
                )
            )

        return patterns

    @override
    def act(self) -> list[Action]:
        actions: list[Action] = []
        for waste in self.identify_waste():
            match waste:
                case EmptyRoomHeating() | OverHeating():
                    actions.append(ReduceHeating(target_device=self.room.id))
        return actions
