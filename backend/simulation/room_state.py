"""Room state - the current physical state of a room."""

from dataclasses import dataclass


@dataclass
class RoomState:
    """Current physical state of a room - what sensors read from."""

    temperature: float  # °C
    co2_ppm: float
    occupied: bool
    heating_power_w: float  # Current heating power consumption
    ventilation_power_w: float  # Current ventilation power consumption

    @property
    def total_hvac_power_w(self) -> float:
        """Total HVAC power consumption."""
        return self.heating_power_w + self.ventilation_power_w
