"""RoomZone - atomic energy zone wrapping a single room."""

from typing import TYPE_CHECKING, override

from core.models import Room
from core.sensors import Sensor, SensorKind
from core.zones.actions import Action
from core.zones.base import EnergyZone
from core.zones.metrics import Metrics
from core.zones.patterns import WastePattern

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
        return []

    @override
    def act(self) -> list[Action]:
        return []
