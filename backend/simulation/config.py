"""Centralised simulation tunables.

Every magic number that controls the physics simulation lives here.
Create a custom ``SimConfig`` to tweak values for testing::

    cfg = SimConfig(forced_power_w=1000.0)
    env = SimulatedEnvironment(room_configs, config=cfg)
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimConfig:
    """All simulation tunables, grouped by category."""

    # --- Tick timing ---
    tick_duration_s: float = 180.0  # 3 minutes per tick

    # --- External environment ---
    external_temp_base_c: float = 3.0  # base outdoor temp
    external_temp_amplitude_c: float = 3.0  # day/night swing

    # --- Solar ---
    peak_irradiance_w_m2: float = 300.0  # peak clear-sky irradiance
    sunrise_hour: float = 9.0
    sunset_hour: float = 15.0
    peak_sun_elevation_deg: float = 15.0

    # --- Occupancy random walk ---
    occupancy_min_ticks: int = 3  # min ticks before state flip
    occupancy_transition_rate: float = 0.15  # multiplier on base probability
    occupancy_min_probability: float = 0.02  # floor on transition probability
    occupancy_heat_gain_w: float = 100.0  # sensible heat per person

    # --- Temperature / CO2 clamps ---
    temp_min_c: float = 5.0
    temp_max_c: float = 50.0
    co2_min_ppm: float = 350.0
    co2_max_ppm: float = 2000.0
    co2_baseline_ppm: float = 400.0

    # --- HVAC defaults (used by HVACConfig) ---
    hvac_max_heating_w: float = 2000.0
    hvac_max_ventilation_w: float = 100.0
    hvac_target_temp_c: float = 21.0
    hvac_heating_gain: float = 200.0  # W per °C below target

    # --- Control action overrides (main.py _apply_actions) ---
    control_target_temp_c: float = 21.0
    control_heating_gain: float = 450.0
    control_max_heating_w: float = 2500.0
    boost_target_temp_c: float = 22.5
    boost_heating_gain: float = 700.0
    boost_max_heating_w: float = 3000.0
    reduce_vent_power_w: float = 30.0
    suspend_target_temp_c: float = 15.0
    suspend_heating_gain: float = 100.0

    # --- System OFF forced power ---
    forced_power_w: float = 500.0

    # --- Per-room starting temperature overrides ---
    # room_id → °C; rooms not listed start at their HVAC target temp
    start_temp_overrides: dict[str, float] = field(default_factory=lambda: {"r-003": 40, "r-105": 23.0})


DEFAULT = SimConfig()
