"""Room physics configuration - constants that define room thermal behavior."""

from dataclasses import dataclass, field


@dataclass
class WallConfig:
    """Physical properties of a wall."""

    neighbor_id: str  # Room ID or "exterior"
    length_m: float
    height_m: float = 3.0
    u_value: float = 1.5  # W/(m²·K) - interior wall default

    @property
    def area_m2(self) -> float:
        return self.length_m * self.height_m

    def heat_transfer_w(self, delta_t: float) -> float:
        """Heat transfer in watts. Positive = heat flowing out through this wall."""
        return self.u_value * self.area_m2 * delta_t


@dataclass
class WindowConfig:
    """A window on an exterior wall."""

    orientation: str  # "N", "S", "E", "W"
    area_m2: float
    transmittance: float = 0.6  # fraction of solar energy passing through


@dataclass
class HVACConfig:
    """HVAC system properties for a room."""

    max_heating_power_w: float = 2000.0
    max_ventilation_power_w: float = 100.0
    target_temperature: float = 21.0
    # Proportional control gain - how aggressively heating responds to temp difference
    heating_gain: float = 200.0  # W per degree below target


@dataclass
class RoomPhysicsConfig:
    """Complete physical configuration for a room."""

    room_id: str
    volume_m3: float
    # Thermal mass: energy needed to change room temp by 1K
    # Approximation: air + furnishings, typically 3-5x air alone
    thermal_mass_j_per_k: float
    walls: list[WallConfig] = field(default_factory=list)
    windows: list[WindowConfig] = field(default_factory=list)
    hvac: HVACConfig = field(default_factory=HVACConfig)
    # CO2 generation rate per occupant (ppm/s at this volume)
    co2_generation_rate: float = 0.005
    # Base CO2 decay rate from natural air exchange (fraction per second)
    co2_decay_rate: float = 0.0001


def estimate_thermal_mass(volume_m3: float, furnishing_factor: float = 3.0) -> float:
    """Estimate thermal mass from room volume.

    Air has ~1200 J/(m³·K). Furnishings add thermal mass.
    furnishing_factor=3.0 means total thermal mass is 3x air alone.
    """
    air_thermal_mass = volume_m3 * 1200  # J/K
    return air_thermal_mass * furnishing_factor
