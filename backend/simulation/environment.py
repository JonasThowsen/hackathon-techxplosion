"""Room environment sources - protocol and implementations."""

from typing import Protocol

from simulation.room_config import RoomPhysicsConfig
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


class SimulatedEnvironment:
    """Physics-based room simulation.

    Simulates heat transfer, HVAC behavior, CO2 levels, and occupancy.
    Supports scenarios for testing and demonstration.
    """

    def __init__(
        self,
        room_configs: dict[str, RoomPhysicsConfig],
        external_temp: float = 10.0,
        tick_duration_s: float = 180.0,  # 3 minutes per tick
    ) -> None:
        self.room_configs = room_configs
        self.external_temp = external_temp
        self.tick_duration_s = tick_duration_s

        # Initialize states
        self.states: dict[str, RoomState] = {}
        for room_id, config in room_configs.items():
            self.states[room_id] = RoomState(
                temperature=config.hvac.target_temperature,
                co2_ppm=400.0,
                occupied=False,
                heating_power_w=0.0,
                ventilation_power_w=0.0,
            )

        # Active scenarios
        self.scenarios: list[ActiveScenario] = []

        # Occupancy schedule (can be overridden by scenarios)
        self._occupancy_seed = 42

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

        # Update each room
        for room_id, config in self.room_configs.items():
            room_scenarios = scenarios_by_room.get(room_id, [])
            self._step_room(room_id, config, tick, room_scenarios)

    def _step_room(
        self,
        room_id: str,
        config: RoomPhysicsConfig,
        tick: int,
        scenarios: list[Scenario],
    ) -> None:
        """Update a single room's state."""
        state = self.states[room_id]

        # Check for force overrides first
        for scenario in scenarios:
            match scenario:
                case ForceTemperature(temperature=temp):
                    state.temperature = temp
                    return  # Skip all other calculations
                case _:
                    pass

        # Determine occupancy
        occupied = self._calculate_occupancy(room_id, tick)
        for scenario in scenarios:
            match scenario:
                case ForceOccupancy(occupied=forced):
                    occupied = forced
                case _:
                    pass
        state.occupied = occupied

        # Calculate heat balance
        heat_delta_j = 0.0

        # Heat loss through walls
        for wall in config.walls:
            neighbor_temp = self._get_neighbor_temp(wall.neighbor_id, scenarios)
            delta_t = state.temperature - neighbor_temp
            heat_loss_w = wall.heat_transfer_w(delta_t)
            heat_delta_j -= heat_loss_w * self.tick_duration_s

        # Extra heat loss from scenarios (window open, door propped)
        for scenario in scenarios:
            match scenario:
                case WindowOpen(heat_loss_w_per_k=loss_rate):
                    ext_temp = self._get_external_temp(scenarios)
                    delta_t = state.temperature - ext_temp
                    heat_delta_j -= loss_rate * delta_t * self.tick_duration_s
                case DoorPropped(neighbor_room_id=neighbor, heat_exchange_w_per_k=rate):
                    if neighbor in self.states:
                        neighbor_temp = self.states[neighbor].temperature
                        delta_t = state.temperature - neighbor_temp
                        heat_delta_j -= rate * delta_t * self.tick_duration_s
                case _:
                    pass

        # Heat gain from occupancy (~80W sensible heat per person, assume 1-2 people)
        if occupied:
            heat_delta_j += 100.0 * self.tick_duration_s

        # Calculate heating power
        heating_power = self._calculate_heating_power(state, config, scenarios)
        state.heating_power_w = heating_power
        heat_delta_j += heating_power * self.tick_duration_s

        # Apply temperature change
        if config.thermal_mass_j_per_k > 0:
            temp_change = heat_delta_j / config.thermal_mass_j_per_k
            state.temperature += temp_change
            # Clamp to reasonable range
            state.temperature = max(5.0, min(35.0, state.temperature))

        # Update CO2
        self._update_co2(state, config, scenarios)

        # Calculate ventilation power
        state.ventilation_power_w = self._calculate_ventilation_power(config, scenarios)

    def _calculate_occupancy(self, room_id: str, tick: int) -> bool:
        """Calculate base occupancy from schedule."""
        import math
        import random

        hour = (tick * 0.05) % 24  # ~3 min per tick
        room_hash = hash(room_id) % 100

        # Simple schedule: more likely occupied during day
        if 8 <= hour < 22:
            base_prob = 0.5
        elif hour >= 22 or hour < 6:
            base_prob = 0.2
        else:
            base_prob = 0.3

        noise = math.sin(tick * 0.1 + room_hash) * 0.15
        rng = random.Random(tick * 100 + room_hash + self._occupancy_seed)
        return rng.random() < (base_prob + noise)

    def _get_neighbor_temp(self, neighbor_id: str, scenarios: list[Scenario]) -> float:
        """Get temperature of a neighboring space."""
        if neighbor_id == "exterior":
            return self._get_external_temp(scenarios)
        if neighbor_id in self.states:
            return self.states[neighbor_id].temperature
        return self.external_temp

    def _get_external_temp(self, scenarios: list[Scenario]) -> float:
        """Get external temperature, considering overrides."""
        for scenario in scenarios:
            match scenario:
                case ExternalTempOverride(temperature=temp):
                    return temp
                case _:
                    pass
        return self.external_temp

    def _calculate_heating_power(
        self,
        state: RoomState,
        config: RoomPhysicsConfig,
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
        temp_error = config.hvac.target_temperature - effective_temp
        if temp_error <= 0:
            return 0.0

        power = temp_error * config.hvac.heating_gain
        return min(power, config.hvac.max_heating_power_w)

    def _update_co2(
        self,
        state: RoomState,
        config: RoomPhysicsConfig,
        scenarios: list[Scenario],
    ) -> None:
        """Update CO2 levels."""
        # CO2 increase from occupancy
        if state.occupied:
            state.co2_ppm += config.co2_generation_rate * self.tick_duration_s

        # CO2 decay from ventilation
        ventilation_working = True
        for scenario in scenarios:
            match scenario:
                case VentilationFailed():
                    ventilation_working = False
                case _:
                    pass

        if ventilation_working:
            decay = config.co2_decay_rate * self.tick_duration_s
            state.co2_ppm -= (state.co2_ppm - 400.0) * decay

        # Clamp
        state.co2_ppm = max(350.0, min(2000.0, state.co2_ppm))

    def _calculate_ventilation_power(
        self,
        config: RoomPhysicsConfig,
        scenarios: list[Scenario],
    ) -> float:
        """Calculate ventilation power consumption."""
        for scenario in scenarios:
            match scenario:
                case VentilationFailed():
                    return 0.0
                case _:
                    pass
        return config.hvac.max_ventilation_power_w
