"""Room environment sources - protocol and implementations."""

import logging
import math
import random
from typing import Protocol

from data.weather import external_temp_at_tick
from simulation.config import DEFAULT, SimConfig
from simulation.room_config import RoomPhysicsConfig, WindowConfig
from simulation.room_state import RoomState
from simulation.scenarios import (
    ActiveScenario,
    DoorPropped,
    ExternalTempOverride,
    ForceHeatingPower,
    ForceOccupancy,
    ForceTemperature,
    HeatingStuckOff,
    HeatingStuckOn,
    Scenario,
    ThermostatStuck,
    VentilationFailed,
    WindowOpen,
    is_scenario_active,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Solar geometry helpers
# ---------------------------------------------------------------------------

# Window normal azimuths in degrees (clockwise from north)
_WINDOW_NORMALS: dict[str, float] = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}


def _tick_to_hour(tick: int, tick_duration_s: float) -> float:
    """Convert tick number to simulated hour of day (0-24)."""
    return (tick * tick_duration_s / 60 / 60) % 24


def _sun_elevation(
    hour: float,
    sunrise: float,
    sunset: float,
    peak_elevation: float,
) -> float:
    """Sun elevation angle in degrees. 0 at sunrise/sunset, peak at noon."""
    if hour < sunrise or hour > sunset:
        return 0.0
    day_length = sunset - sunrise
    return peak_elevation * math.sin(math.pi * (hour - sunrise) / day_length)


def _sun_azimuth(hour: float, sunrise: float, sunset: float) -> float:
    """Sun azimuth in degrees. Rises East (90°), noon South (180°), sets West (270°)."""
    if hour < sunrise or hour > sunset:
        return 180.0  # arbitrary, elevation is 0 anyway
    frac = (hour - sunrise) / (sunset - sunrise)
    return 90.0 + frac * 180.0


def _solar_irradiance(elevation: float, peak_irradiance: float) -> float:
    """Direct beam irradiance on a surface perpendicular to sun rays (W/m²)."""
    if elevation <= 0:
        return 0.0
    return peak_irradiance * math.sin(math.radians(elevation))


def _solar_gain_for_window(window: WindowConfig, hour: float, cfg: SimConfig) -> float:
    """Calculate solar heat gain through a single window in watts."""
    elevation = _sun_elevation(hour, cfg.sunrise_hour, cfg.sunset_hour, cfg.peak_sun_elevation_deg)
    if elevation <= 0:
        return 0.0

    azimuth = _sun_azimuth(hour, cfg.sunrise_hour, cfg.sunset_hour)
    irradiance = _solar_irradiance(elevation, cfg.peak_irradiance_w_m2)

    # Angle between sun azimuth and window outward normal
    normal_az = _WINDOW_NORMALS.get(window.orientation, 180.0)
    incidence_h = math.radians(azimuth - normal_az)

    # Incidence factor: cos(horizontal angle) * cos(elevation) gives
    # approximate fraction of direct beam hitting the vertical window surface.
    # Simplified: for a vertical window the effective irradiance is
    # irradiance * cos(incidence_horizontal).
    cos_inc = math.cos(incidence_h)
    if cos_inc <= 0:
        return 0.0

    return window.area_m2 * irradiance * window.transmittance * cos_inc


# ---------------------------------------------------------------------------
# Occupancy random-walk helpers
# ---------------------------------------------------------------------------

# Room type classification based on room name keywords
_ROOM_TYPE_KEYWORDS: dict[str, list[str]] = {
    "bedroom": ["bedroom"],
    "kitchen": ["kitchen"],
    "common": ["common room"],
    "study": ["study"],
    "lobby": ["lobby"],
    "storage": ["storage"],
    "laundry": ["laundry"],
    "bathroom": ["bathroom"],
}


def _classify_room(room_name: str) -> str:
    """Classify a room by its name into a type key."""
    lower = room_name.lower()
    for room_type, keywords in _ROOM_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return room_type
    return "other"


def _base_occupancy_probability(room_type: str, hour: float) -> float:
    """Target occupancy probability by room type and hour of day."""
    match room_type:
        case "bedroom":
            # High at night, low during day
            if hour >= 22 or hour < 8:
                return 0.85
            if 8 <= hour < 10:
                return 0.3
            return 0.1
        case "kitchen":
            # Peaks at meal times
            if 7 <= hour < 9:
                return 0.6
            if 12 <= hour < 13:
                return 0.5
            if 18 <= hour < 20:
                return 0.7
            return 0.05
        case "common" | "study":
            # Active during daytime and evening
            if 10 <= hour < 22:
                return 0.5
            return 0.05
        case "lobby":
            if 8 <= hour < 22:
                return 0.15
            return 0.02
        case "bathroom":
            # Brief spikes, moderate baseline during waking hours
            if 7 <= hour < 9:
                return 0.4
            if hour >= 22 or hour < 6:
                return 0.05
            return 0.15
        case "storage" | "laundry":
            if 10 <= hour < 20:
                return 0.1
            return 0.02
        case _:
            return 0.1


class _OccupancyState:
    """Per-room random-walk occupancy state."""

    __slots__ = ("_min_prob", "_min_ticks", "_rate", "occupied", "rng", "ticks_in_state")

    def __init__(self, seed: int, cfg: SimConfig) -> None:
        self.occupied: bool = False
        self.ticks_in_state: int = 0
        self.rng: random.Random = random.Random(seed)
        self._min_ticks: int = cfg.occupancy_min_ticks
        self._rate: float = cfg.occupancy_transition_rate
        self._min_prob: float = cfg.occupancy_min_probability

    def step(self, target_prob: float) -> bool:
        """Advance one tick. Returns new occupancy status."""
        self.ticks_in_state += 1

        if self.ticks_in_state < self._min_ticks:
            return self.occupied

        # Transition probability nudged toward the target
        if self.occupied:
            leave_prob = max(self._min_prob, (1.0 - target_prob) * self._rate)
            if self.rng.random() < leave_prob:
                self.occupied = False
                self.ticks_in_state = 0
        else:
            enter_prob = max(self._min_prob, target_prob * self._rate)
            if self.rng.random() < enter_prob:
                self.occupied = True
                self.ticks_in_state = 0

        return self.occupied


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class RoomEnvironmentSource(Protocol):
    """Protocol for room environment data sources.

    Implementations can provide real sensor data or simulated data.
    Sensors read from this without knowing the source.
    """

    def get_state(self, room_id: str) -> RoomState:
        """Get current state of a room."""
        ...

    def step(self, tick: int) -> None:
        """Advance the environment by one tick."""
        ...


# ---------------------------------------------------------------------------
# Simulated environment
# ---------------------------------------------------------------------------


class SimulatedEnvironment:
    """Physics-based room simulation.

    Simulates heat transfer, HVAC behavior, CO2 levels, and occupancy.
    Supports scenarios for testing and demonstration.
    """

    def __init__(
        self,
        room_configs: dict[str, RoomPhysicsConfig],
        room_names: dict[str, str] | None = None,
        config: SimConfig = DEFAULT,
    ) -> None:
        self.room_configs = room_configs
        self.config = config

        # Initialize states
        self.states: dict[str, RoomState] = {}
        for room_id, rc in room_configs.items():
            start_temp = config.start_temp_overrides.get(room_id, rc.hvac.target_temperature)
            self.states[room_id] = RoomState(
                temperature=start_temp,
                co2_ppm=config.co2_baseline_ppm,
                occupied=False,
                heating_power_w=0.0,
                ventilation_power_w=0.0,
            )

        # Active scenarios
        self.scenarios: list[ActiveScenario] = []

        # Whether solar gain is enabled (can be toggled for testing)
        self.sun_enabled: bool = True

        # Occupancy random-walk state per room
        names = room_names or {}
        self._room_types: dict[str, str] = {}
        self._occupancy: dict[str, _OccupancyState] = {}
        for i, room_id in enumerate(sorted(room_configs)):
            name = names.get(room_id, room_id)
            self._room_types[room_id] = _classify_room(name)
            self._occupancy[room_id] = _OccupancyState(seed=42 + i, cfg=config)

    def add_scenario(
        self,
        room_id: str,
        scenario: Scenario,
        start_tick: int | None = None,
        end_tick: int | None = None,
    ) -> None:
        """Add a scenario to the simulation."""
        self.scenarios.append(
            ActiveScenario(
                room_id=room_id,
                scenario=scenario,
                start_tick=start_tick,
                end_tick=end_tick,
            )
        )

    def clear_scenarios(self, room_id: str | None = None) -> None:
        """Clear scenarios, optionally for a specific room only."""
        if room_id is None:
            self.scenarios.clear()
        else:
            self.scenarios = [s for s in self.scenarios if s.room_id != room_id]

    def get_state(self, room_id: str) -> RoomState:
        """Get current state of a room."""
        return self.states[room_id]

    def step(self, tick: int) -> None:
        """Advance simulation by one tick."""
        # Collect active scenarios per room
        scenarios_by_room: dict[str, list[Scenario]] = {}
        for active in self.scenarios:
            if is_scenario_active(active, tick):
                scenarios_by_room.setdefault(active.room_id, []).append(active.scenario)

        # Derive hour from tick
        hour = _tick_to_hour(tick, self.config.tick_duration_s)

        # Update each room
        for room_id, rc in self.room_configs.items():
            room_scenarios = scenarios_by_room.get(room_id, [])
            self._step_room(room_id, rc, tick, hour, room_scenarios)

    def _step_room(
        self,
        room_id: str,
        room_cfg: RoomPhysicsConfig,
        tick: int,
        hour: float,
        scenarios: list[Scenario],
    ) -> None:
        """Update a single room's state."""
        state = self.states[room_id]
        cfg = self.config

        # Check for force overrides first
        for scenario in scenarios:
            match scenario:
                case ForceTemperature(temperature=temp):
                    state.temperature = temp
                    return  # Skip all other calculations
                case _:
                    pass

        # Determine occupancy (random walk)
        occupied = self._calculate_occupancy(room_id, hour)
        for scenario in scenarios:
            match scenario:
                case ForceOccupancy(occupied=forced):
                    occupied = forced
                case _:
                    pass
        state.occupied = occupied

        # Calculate heat balance
        heat_delta_j = 0.0

        # Time-varying external temperature from real yr.no data
        ext_temp_base = external_temp_at_tick(tick, cfg.tick_duration_s)

        # Heat loss through walls
        for wall in room_cfg.walls:
            neighbor_temp = self._get_neighbor_temp(wall.neighbor_id, ext_temp_base, scenarios)
            delta_t = state.temperature - neighbor_temp
            heat_loss_w = wall.heat_transfer_w(delta_t)
            heat_delta_j -= heat_loss_w * cfg.tick_duration_s
            # if abs(delta_t) > 0.5:
            #     logger.debug(
            #         "Heat transfer: %s -> %s | ΔT=%.2f°C | Q=%.1fW | wall=%.1fm²",
            #         room_id,
            #         wall.neighbor_id,
            #         delta_t,
            #         heat_loss_w,
            #         wall.area_m2,
            #     )

        # Solar gain through windows
        if self.sun_enabled:
            for window in room_cfg.windows:
                gain_w = _solar_gain_for_window(window, hour, cfg)
                heat_delta_j += gain_w * cfg.tick_duration_s

        # Extra heat loss from scenarios (window open, door propped)
        for scenario in scenarios:
            match scenario:
                case WindowOpen(heat_loss_w_per_k=loss_rate):
                    ext_temp = self._get_external_temp(ext_temp_base, scenarios)
                    delta_t = state.temperature - ext_temp
                    heat_delta_j -= loss_rate * delta_t * cfg.tick_duration_s
                case DoorPropped(neighbor_room_id=neighbor, heat_exchange_w_per_k=rate):
                    if neighbor in self.states:
                        neighbor_temp = self.states[neighbor].temperature
                        delta_t = state.temperature - neighbor_temp
                        heat_delta_j -= rate * delta_t * cfg.tick_duration_s
                case _:
                    pass

        # Heat gain from occupancy
        if occupied:
            heat_delta_j += cfg.occupancy_heat_gain_w * cfg.tick_duration_s

        # Calculate heating power
        heating_power = self._calculate_heating_power(state, room_cfg, scenarios)
        state.heating_power_w = heating_power
        heat_delta_j += heating_power * cfg.tick_duration_s

        # Apply temperature change
        if room_cfg.thermal_mass_j_per_k > 0:
            temp_change = heat_delta_j / room_cfg.thermal_mass_j_per_k
            state.temperature += temp_change
            # Clamp to reasonable range
            state.temperature = max(cfg.temp_min_c, min(cfg.temp_max_c, state.temperature))
            # if abs(temp_change) > 0.01:
            #     logger.debug(
            #         "Room %s: temp %.2f°C -> %.2f°C (Δ%.3f°C, heat_delta=%.0fJ)",
            #         room_id,
            #         state.temperature - temp_change,
            #         state.temperature,
            #         temp_change,
            #         heat_delta_j,
            #     )

        # Update CO2
        self._update_co2(state, room_cfg, scenarios)

        # Calculate ventilation power
        state.ventilation_power_w = self._calculate_ventilation_power(room_cfg, scenarios)

    def _calculate_occupancy(self, room_id: str, hour: float) -> bool:
        """Calculate occupancy using per-room random walk."""
        room_type = self._room_types.get(room_id, "other")
        target = _base_occupancy_probability(room_type, hour)
        occ_state = self._occupancy[room_id]
        return occ_state.step(target)

    def _get_neighbor_temp(self, neighbor_id: str, ext_temp_base: float, scenarios: list[Scenario]) -> float:
        """Get temperature of a neighboring space."""
        if neighbor_id == "exterior":
            return self._get_external_temp(ext_temp_base, scenarios)
        if neighbor_id in self.states:
            return self.states[neighbor_id].temperature
        return ext_temp_base

    def _get_external_temp(self, ext_temp_base: float, scenarios: list[Scenario]) -> float:
        """Get external temperature, considering overrides."""
        for scenario in scenarios:
            match scenario:
                case ExternalTempOverride(temperature=temp):
                    return temp
                case _:
                    pass
        return ext_temp_base

    def _calculate_heating_power(
        self,
        state: RoomState,
        room_cfg: RoomPhysicsConfig,
        scenarios: list[Scenario],
    ) -> float:
        """Calculate current heating power based on HVAC config and scenarios."""
        # Check for heating overrides
        for scenario in scenarios:
            match scenario:
                case HeatingStuckOn(power_w=power):
                    return power
                case HeatingStuckOff():
                    return 0.0
                case ForceHeatingPower(power_w=power):
                    return power
                case _:
                    pass

        # Get effective temperature reading (may be affected by stuck thermostat)
        effective_temp = state.temperature
        for scenario in scenarios:
            match scenario:
                case ThermostatStuck(stuck_reading=reading):
                    effective_temp = reading
                case _:
                    pass

        # Simple proportional control
        temp_error = room_cfg.hvac.target_temperature - effective_temp
        if temp_error <= 0:
            return 0.0

        power = temp_error * room_cfg.hvac.heating_gain
        return min(power, room_cfg.hvac.max_heating_power_w)

    def _update_co2(
        self,
        state: RoomState,
        room_cfg: RoomPhysicsConfig,
        scenarios: list[Scenario],
    ) -> None:
        """Update CO2 levels."""
        cfg = self.config

        # CO2 increase from occupancy
        if state.occupied:
            state.co2_ppm += room_cfg.co2_generation_rate * cfg.tick_duration_s

        # CO2 decay from ventilation
        ventilation_working = True
        for scenario in scenarios:
            match scenario:
                case VentilationFailed():
                    ventilation_working = False
                case _:
                    pass

        if ventilation_working:
            decay = room_cfg.co2_decay_rate * cfg.tick_duration_s
            state.co2_ppm -= (state.co2_ppm - cfg.co2_baseline_ppm) * decay

        # Clamp
        state.co2_ppm = max(cfg.co2_min_ppm, min(cfg.co2_max_ppm, state.co2_ppm))

    def _calculate_ventilation_power(
        self,
        room_cfg: RoomPhysicsConfig,
        scenarios: list[Scenario],
    ) -> float:
        """Calculate ventilation power consumption."""
        for scenario in scenarios:
            match scenario:
                case VentilationFailed():
                    return 0.0
                case _:
                    pass
        return room_cfg.hvac.max_ventilation_power_w
