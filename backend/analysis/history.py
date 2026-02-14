"""Historical metrics storage for rooms."""

from dataclasses import dataclass, field
from datetime import datetime

from core.models import RoomMetrics


@dataclass
class RoomMetricsSnapshot:
    timestamp: datetime
    metrics: RoomMetrics


@dataclass
class RoomHistory:
    room_id: str

    timestamps: list[datetime] = field(default_factory=list)
    temperature: list[float] = field(default_factory=list)
    co2: list[float] = field(default_factory=list)
    heating_power: list[float] = field(default_factory=list)
    ventilation_power: list[float] = field(default_factory=list)

    def add(self, timestamp: datetime, metrics: RoomMetrics) -> None:
        self.timestamps.append(timestamp)
        self.temperature.append(metrics.temperature)
        self.co2.append(metrics.co2)
        self.heating_power.append(metrics.heating_power)
        self.ventilation_power.append(metrics.ventilation_power)
