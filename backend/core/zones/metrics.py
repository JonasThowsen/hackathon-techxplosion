"""Aggregated metrics at any level of the zone hierarchy."""

from dataclasses import dataclass


@dataclass
class Metrics:
    """Aggregated metrics at any level of the zone hierarchy."""

    temperature: float
    occupancy: bool
    co2: float
    heating_power: float
    ventilation_power: float

    @property
    def total_hvac_power(self) -> float:
        return self.heating_power + self.ventilation_power
