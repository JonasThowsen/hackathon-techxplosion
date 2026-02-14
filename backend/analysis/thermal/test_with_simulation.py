"""Test thermal estimation against the actual simulation system.

This test runs the SimulatedEnvironment and then attempts to recover
the true thermal parameters from the simulated data.
"""

from datetime import datetime, timedelta

from analysis.thermal import (
    EdgeSpec,
    EstimationConfig,
    ExternalConditions,
    RoomTimeSeries,
    ThermalGraph,
    assess_building_data_quality,
    check_estimation_readiness,
    estimate_parameters,
)
from simulation.config import SimConfig
from simulation.environment import SimulatedEnvironment
from simulation.room_config import (
    HVACConfig,
    RoomPhysicsConfig,
    WallConfig,
    WindowConfig,
)
from simulation.scenarios import HeatingStuckOff, WindowOpen


def build_test_building() -> tuple[dict[str, RoomPhysicsConfig], dict[str, str]]:
    """Build a simple test building with known thermal properties.

    Layout (single floor):
        [Room A] -- [Room B] -- [Room C]
           |                       |
        exterior                exterior

    Returns:
        Tuple of (room_configs, room_names)
    """
    # Define thermal properties we want to recover
    # Using realistic values that the simulator uses

    room_a = RoomPhysicsConfig(
        room_id="room_a",
        volume_m3=30.0,  # 10m^2 x 3m
        thermal_mass_j_per_k=108_000.0,  # 30 * 1200 * 3 (air + furnishings)
        walls=[
            WallConfig(neighbor_id="room_b", length_m=3.0, height_m=3.0, u_value=1.5),
            WallConfig(neighbor_id="exterior", length_m=4.0, height_m=3.0, u_value=0.5),
            WallConfig(neighbor_id="exterior", length_m=3.0, height_m=3.0, u_value=0.5),
        ],
        windows=[
            WindowConfig(orientation="S", area_m2=2.0, transmittance=0.6),
        ],
        hvac=HVACConfig(max_heating_power_w=2000.0, target_temperature=21.0, heating_gain=200.0),
    )

    room_b = RoomPhysicsConfig(
        room_id="room_b",
        volume_m3=45.0,  # 15m^2 x 3m (interior room, larger)
        thermal_mass_j_per_k=162_000.0,
        walls=[
            WallConfig(neighbor_id="room_a", length_m=3.0, height_m=3.0, u_value=1.5),
            WallConfig(neighbor_id="room_c", length_m=3.0, height_m=3.0, u_value=1.5),
        ],
        windows=[],  # No exterior walls
        hvac=HVACConfig(max_heating_power_w=2000.0, target_temperature=21.0, heating_gain=200.0),
    )

    room_c = RoomPhysicsConfig(
        room_id="room_c",
        volume_m3=36.0,  # 12m^2 x 3m
        thermal_mass_j_per_k=129_600.0,
        walls=[
            WallConfig(neighbor_id="room_b", length_m=3.0, height_m=3.0, u_value=1.5),
            WallConfig(neighbor_id="exterior", length_m=4.0, height_m=3.0, u_value=0.5),
            WallConfig(neighbor_id="exterior", length_m=3.0, height_m=3.0, u_value=0.5),
        ],
        windows=[
            WindowConfig(orientation="S", area_m2=1.5, transmittance=0.6),
        ],
        hvac=HVACConfig(max_heating_power_w=2000.0, target_temperature=21.0, heating_gain=200.0),
    )

    configs = {
        "room_a": room_a,
        "room_b": room_b,
        "room_c": room_c,
    }

    names = {
        "room_a": "Study",  # Occupied during day
        "room_b": "Common Room",
        "room_c": "Bedroom",  # Occupied at night
    }

    return configs, names


def extract_true_parameters(
    configs: dict[str, RoomPhysicsConfig],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Extract the true thermal parameters from room configs.

    Returns:
        Tuple of (thermal_mass_dict, conductance_dict)
    """
    thermal_mass: dict[str, float] = {}
    conductance: dict[tuple[str, str], float] = {}

    for room_id, cfg in configs.items():
        thermal_mass[room_id] = cfg.thermal_mass_j_per_k

        for wall in cfg.walls:
            g = wall.u_value * wall.area_m2

            neighbor = wall.neighbor_id
            if neighbor == "exterior":
                key: tuple[str, str] = (room_id, "exterior")
            else:
                key = tuple(sorted([room_id, neighbor]))  # type: ignore[assignment]

            if key not in conductance:
                conductance[key] = g

    return thermal_mass, conductance


def build_thermal_graph(configs: dict[str, RoomPhysicsConfig]) -> ThermalGraph:
    """Build ThermalGraph from room configs."""
    room_ids = list(configs.keys())
    edges: list[EdgeSpec] = []
    seen: set[tuple[str, str]] = set()

    for room_id, cfg in configs.items():
        for wall in cfg.walls:
            if wall.neighbor_id == "exterior":
                edges.append(
                    EdgeSpec(
                        node_a=room_id,
                        node_b="exterior",
                        wall_area_m2=wall.area_m2,
                        is_exterior=True,
                    )
                )
            else:
                pair = sorted([room_id, wall.neighbor_id])
                key: tuple[str, str] = (pair[0], pair[1])
                if key not in seen:
                    seen.add(key)
                    edges.append(
                        EdgeSpec(
                            node_a=key[0],
                            node_b=key[1],
                            wall_area_m2=wall.area_m2,
                            is_exterior=False,
                        )
                    )

    return ThermalGraph(node_ids=room_ids, edges=edges)


def run_simulation(
    configs: dict[str, RoomPhysicsConfig],
    names: dict[str, str],
    n_days: int = 14,
    sun_enabled: bool = True,
    add_excitation: bool = False,
) -> tuple[dict[str, RoomTimeSeries], list[float], SimConfig]:
    """Run the simulation and collect data.

    Args:
        add_excitation: If True, add scenarios that force temperature variation
                       (heating failures, window open events)

    Returns:
        Tuple of (room_histories, external_temps, sim_config)
    """
    sim_config = SimConfig(
        tick_duration_s=3600.0,  # one hour
        external_temp_base_c=5.0,
        external_temp_amplitude_c=6.0,  # Bigger day/night swing
    )

    env = SimulatedEnvironment(
        room_configs=configs,
        room_names=names,
        config=sim_config,
    )
    env.sun_enabled = sun_enabled

    # Add excitation scenarios if requested
    if add_excitation:
        ticks_per_hour = int(3600 / sim_config.tick_duration_s)
        ticks_per_day = 24 * ticks_per_hour

        # Night setback: heating off from 11pm to 6am every night
        for day in range(n_days):
            night_start = day * ticks_per_day + 23 * ticks_per_hour
            night_end = day * ticks_per_day + 30 * ticks_per_hour  # 6am next day

            for room_id in configs:
                env.add_scenario(room_id, HeatingStuckOff(), night_start, night_end)

        # Window open events (creates identifiable heat loss spikes)
        # Day 3: room_a window open for 2 hours mid-day
        env.add_scenario(
            "room_a",
            WindowOpen(heat_loss_w_per_k=30.0),
            3 * ticks_per_day + 12 * ticks_per_hour,
            3 * ticks_per_day + 14 * ticks_per_hour,
        )

        # Day 6: room_c window open for 3 hours
        env.add_scenario(
            "room_c",
            WindowOpen(heat_loss_w_per_k=25.0),
            6 * ticks_per_day + 10 * ticks_per_hour,
            6 * ticks_per_day + 13 * ticks_per_hour,
        )

        # Day 10: room_b brief heating failure (4 hours)
        env.add_scenario(
            "room_b",
            HeatingStuckOff(),
            10 * ticks_per_day + 8 * ticks_per_hour,
            10 * ticks_per_day + 12 * ticks_per_hour,
        )

    # Calculate number of ticks
    ticks_per_hour = 3600.0 / sim_config.tick_duration_s
    n_ticks = int(n_days * 24 * ticks_per_hour)

    # Initialize histories
    histories: dict[str, RoomTimeSeries] = {
        room_id: RoomTimeSeries(
            room_id=room_id,
            timestamps=[],
            temperature=[],
            heating_power=[],
        )
        for room_id in configs
    }

    external_temps: list[float] = []
    start_time = datetime(2024, 1, 15, 0, 0, 0)

    print(f"Running simulation: {n_days} days, {n_ticks} ticks...")

    for tick in range(n_ticks):
        # Record state before step
        timestamp = start_time + timedelta(seconds=tick * sim_config.tick_duration_s)

        # Get external temp
        hour = (tick * sim_config.tick_duration_s / 3600) % 24
        ext_temp = sim_config.external_temp_base_c + sim_config.external_temp_amplitude_c * (
            __import__("math").sin(2 * 3.14159 * (hour - 6) / 24)
        )
        external_temps.append(ext_temp)

        for room_id in configs:
            state = env.get_state(room_id)
            histories[room_id].timestamps.append(timestamp)
            histories[room_id].temperature.append(state.temperature)
            histories[room_id].heating_power.append(state.heating_power_w)

        # Step simulation
        env.step(tick)

    return histories, external_temps, sim_config


def run_test(sun_enabled: bool = True, add_excitation: bool = False) -> None:
    """Run the full test."""
    mode_parts: list[str] = []
    if sun_enabled:
        mode_parts.append("solar")
    if add_excitation:
        mode_parts.append("excitation")
    mode = " + ".join(mode_parts) if mode_parts else "baseline"

    print("=" * 70)
    print(f"THERMAL ESTIMATION TEST ({mode})")
    print("=" * 70)

    # Build test building
    configs, names = build_test_building()

    # Extract true parameters
    true_mass, true_cond = extract_true_parameters(configs)

    print("\nTrue parameters:")
    print("  Thermal mass (kJ/K):")
    for room_id, mass in true_mass.items():
        print(f"    {room_id}: {mass / 1000:.1f}")
    print("  Conductance (W/K):")
    for edge, g in true_cond.items():
        print(f"    {edge[0]} <-> {edge[1]}: {g:.1f}")

    # Run simulation
    histories, ext_temps, sim_config = run_simulation(
        configs, names, n_days=14, sun_enabled=sun_enabled, add_excitation=add_excitation
    )

    # Show data summary
    print("\nData summary:")
    for room_id, hist in histories.items():
        temps = hist.temperature
        powers = hist.heating_power
        print(f"  {room_id}: T=[{min(temps):.1f}, {max(temps):.1f}]°C, P=[{min(powers):.0f}, {max(powers):.0f}]W")

    # Assess data quality
    dt_seconds = sim_config.tick_duration_s
    quality = assess_building_data_quality(histories, dt_seconds)
    readiness = check_estimation_readiness(quality)
    print(f"\nData quality: {readiness.quality_summary}")

    # Build graph and run estimation
    graph = build_thermal_graph(configs)

    print("\nRunning estimation...")
    result = estimate_parameters(
        room_data=histories,
        graph=graph,
        external=ExternalConditions(temperature=ext_temps),
        config=EstimationConfig(
            dt_seconds=dt_seconds,
            regularization=0.01,
            loss="huber",
        ),
    )

    # Results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Success: {result.success}")
    print(f"  R-squared: {result.fit_metrics.r_squared:.4f}")
    print(f"  RMSE: {result.fit_metrics.rmse:.4f} K")

    # Compare thermal mass
    print("\nThermal mass estimation:")
    print(f"  {'Room':<10} {'Estimated':>12} {'True':>12} {'Error':>10}")
    print(f"  {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 10}")
    mass_errors: list[float] = []
    for room_id in configs:
        est = result.parameters.rooms[room_id].thermal_mass / 1000
        true = true_mass[room_id] / 1000
        err = 100 * (est - true) / true
        mass_errors.append(abs(err))
        print(f"  {room_id:<10} {est:>12.1f} {true:>12.1f} {err:>+10.1f}%")

    print(f"\n  Mean |error|: {sum(mass_errors) / len(mass_errors):.1f}%")

    # Compare conductance
    print("\nConductance estimation:")
    print(f"  {'Edge':<25} {'Estimated':>10} {'True':>10} {'Error':>10}")
    cond_errors: list[float] = []
    for edge, true_g in true_cond.items():
        if edge[1] == "exterior":
            room_params = result.parameters.rooms.get(edge[0])
            est_g = room_params.exterior_conductance if room_params else 0.0
        else:
            est_g = result.parameters.get_conductance(edge[0], edge[1])

        if est_g is not None and true_g > 0:
            err = 100 * (est_g - true_g) / true_g
            cond_errors.append(abs(err))
            print(f"  {edge[0]:<10}->{edge[1]:<12} {est_g:>10.1f} {true_g:>10.1f} {err:>+10.1f}%")

    if cond_errors:
        print(f"\n  Mean |error|: {sum(cond_errors) / len(cond_errors):.1f}%")

    print("\n" + "=" * 70)


def main() -> None:
    """Run tests with different configurations."""
    # Test 1: Baseline (no solar, no excitation) - should be hard
    run_test(sun_enabled=False, add_excitation=False)

    print("\n\n")

    # Test 2: With excitation (night setback, window events) - should be better
    run_test(sun_enabled=False, add_excitation=True)

    print("\n\n")

    # Test 3: Excitation + solar - realistic scenario
    run_test(sun_enabled=True, add_excitation=True)


if __name__ == "__main__":
    main()
