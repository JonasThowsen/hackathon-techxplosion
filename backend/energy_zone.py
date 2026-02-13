from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Metrics:
    temperature: float
    occupancy: float  # 0.0-1.0 percentage
    co2: float
    power: float


@dataclass
class WastePattern:
    pattern_id: str
    description: str
    estimated_kwh_wasted: float
    duration_minutes: float
    cause: str
    suggested_action: str


@dataclass
class Action:
    action_id: str
    description: str
    target_device: str
    action_type: str  # "reduce_heating", "cut_power", "reduce_ventilation", etc.


class EnergyZone(ABC):
    """Every level of the building hierarchy implements this interface."""

    @abstractmethod
    def collect_metrics(self) -> Metrics: ...

    @abstractmethod
    def identify_waste(self) -> list[WastePattern]: ...

    @abstractmethod
    def act(self) -> list[Action]: ...
