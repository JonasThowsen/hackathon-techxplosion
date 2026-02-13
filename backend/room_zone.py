from energy_zone import Action, EnergyZone, Metrics, WastePattern
from models import Room
from sensors import Sensor, SensorKind


class RoomZone(EnergyZone):
    """Atomic energy zone wrapping a single room and its sensors."""

    def __init__(self, room: Room, sensors: list[Sensor]) -> None:
        self.room = room
        self.sensors = sensors
        self._last_metrics: Metrics | None = None

    def _readings_by_kind(self, kind: SensorKind) -> list[float]:
        return [s.value for s in self.sensors if s.kind == kind]

    def collect_metrics(self) -> Metrics:
        temps = self._readings_by_kind(SensorKind.TEMPERATURE)
        occ = self._readings_by_kind(SensorKind.OCCUPANCY)
        co2 = self._readings_by_kind(SensorKind.CO2)
        power = self._readings_by_kind(SensorKind.POWER)

        metrics = Metrics(
            temperature=sum(temps) / len(temps) if temps else 0.0,
            occupancy=sum(1.0 for v in occ if v > 0.5) / len(occ) if occ else 0.0,
            co2=sum(co2) / len(co2) if co2 else 0.0,
            power=sum(power),
        )
        self._last_metrics = metrics
        return metrics

    def identify_waste(self) -> list[WastePattern]:
        m = self._last_metrics
        if m is None:
            return []

        patterns: list[WastePattern] = []

        if m.occupancy < 0.1 and m.temperature > 22.0 and m.power > 100.0:
            patterns.append(
                WastePattern(
                    pattern_id="empty_room_heating_on",
                    description=f"{self.room.name}: heating on in empty room",
                    estimated_kwh_wasted=m.power * 0.001,
                    duration_minutes=5.0,
                    cause="HVAC running while room unoccupied",
                    suggested_action="reduce_heating",
                )
            )

        if m.temperature > 24.0:
            patterns.append(
                WastePattern(
                    pattern_id="over_heating",
                    description=f"{self.room.name}: temperature above 24°C",
                    estimated_kwh_wasted=(m.temperature - 22.0) * 0.05,
                    duration_minutes=5.0,
                    cause="Temperature setpoint too high",
                    suggested_action="reduce_heating",
                )
            )

        if m.occupancy < 0.1 and m.power > 50.0:
            patterns.append(
                WastePattern(
                    pattern_id="appliances_standby",
                    description=f"{self.room.name}: standby power draw in empty room",
                    estimated_kwh_wasted=m.power * 0.0005,
                    duration_minutes=5.0,
                    cause="Appliances left on standby",
                    suggested_action="cut_power",
                )
            )

        return patterns

    def act(self) -> list[Action]:
        actions: list[Action] = []
        for waste in self.identify_waste():
            actions.append(
                Action(
                    action_id=f"{self.room.id}:{waste.pattern_id}",
                    description=waste.description,
                    target_device=self.room.id,
                    action_type=waste.suggested_action,
                )
            )
        return actions
