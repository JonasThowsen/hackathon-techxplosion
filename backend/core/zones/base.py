"""EnergyZone abstract base class."""

from abc import ABC, abstractmethod

from core.zones.actions import Action
from core.zones.metrics import Metrics
from core.zones.patterns import WastePattern


class EnergyZone(ABC):
    """Every level of the building hierarchy implements this interface."""

    @abstractmethod
    def collect_metrics(self) -> Metrics: ...

    @abstractmethod
    def identify_waste(self) -> list[WastePattern]: ...

    @abstractmethod
    def act(self) -> list[Action]: ...
