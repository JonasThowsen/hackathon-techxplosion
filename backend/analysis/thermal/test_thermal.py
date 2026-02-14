"""Test thermal system identification with synthetic data.

This test demonstrates the full workflow:
1. Generate synthetic data with known parameters
2. Assess data quality
3. Estimate parameters
4. Detect anomalies
5. Analyze results
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from analysis.thermal import (
    AnomalyDetectionConfig,
    EstimationConfig,
    ExternalConditions,
    # Core types
    RoomTimeSeries,
    ThermalGraph,
    analyze_identifiability,
    # Data quality
    assess_building_data_quality,
    # Graph
    build_graph_from_adjacency,
    check_estimation_readiness,
    # Heat flow
    compute_heat_flow_snapshot,
    # Anomaly detection
    detect_anomalies_from_residuals,
    detect_sustained_events,
    # Estimation
    estimate_parameters,
    # Cooling curve
    find_cooling_segments,
    identify_thermal_bridges,
    summarize_anomalies,
)

# -----------------------------------------------------------------------------
# Synthetic Data Generation
# -----------------------------------------------------------------------------


@dataclass
class WindowOpenEvent:
    """Simulated window/door left open event."""

    room_id: str
    start_timestep: int
    duration_timesteps: int
    extra_conductance: float  # W/K


def generate_external_temperature(
    t: int,
    dt_seconds: float,
    base_temp: float = 5.0,
) -> float:
    """Generate realistic external temperature with daily variation."""
    hours_elapsed = (t * dt_seconds) / 3600
    hour_of_day = hours_elapsed % 24
    day = int(hours_elapsed // 24)

    # Daily cycle: coldest at 6am, warmest at 3pm
    daily_variation = 4.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)

    # Day-to-day variation
    rng = np.random.default_rng(42 + day)
    day_offset = rng.uniform(-3, 3)

    return base_temp + daily_variation + day_offset


def simulate_building(
    true_thermal_mass: dict[str, float],
    true_conductance: dict[tuple[str, str], float],
    graph: ThermalGraph,
    n_days: int = 14,
    measurements_per_hour: int = 4,
    base_external_temp: float = 5.0,
    noise_std: float = 0.02,
    window_events: list[WindowOpenEvent] | None = None,
    heating_enabled: bool = True,
    initial_temps: dict[str, float] | None = None,
) -> tuple[dict[str, RoomTimeSeries], list[float]]:
    """Simulate building thermal dynamics with known parameters.

    Returns room histories and external temperature series.
    """
    rng = np.random.default_rng(42)
    window_events = window_events or []

    dt_seconds = 3600.0 / measurements_per_hour
    n_timesteps = n_days * 24 * measurements_per_hour

    # Initialize temperatures
    if initial_temps:
        temperatures = dict(initial_temps)
    else:
        temperatures = {node_id: 15.0 + rng.uniform(-1, 1) for node_id in graph.node_ids}

    # Heating parameters
    target_temp = 21.0
    heating_gain = 300.0  # W per degree below target
    max_power = 2500.0

    # Initialize histories
    histories: dict[str, RoomTimeSeries] = {}
    for node_id in graph.node_ids:
        histories[node_id] = RoomTimeSeries(
            room_id=node_id,
            timestamps=[],
            temperature=[],
            heating_power=[],
        )

    external_temps: list[float] = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    # Simulate
    for t in range(n_timesteps):
        timestamp = start_time + timedelta(seconds=t * dt_seconds)
        ext_temp = generate_external_temperature(t, dt_seconds, base_external_temp)
        external_temps.append(ext_temp)

        # Record state
        for node_id in graph.node_ids:
            histories[node_id].timestamps.append(timestamp)
            histories[node_id].temperature.append(temperatures[node_id])

        # Calculate heating power
        heating_powers: dict[str, float] = {}
        for node_id in graph.node_ids:
            if not heating_enabled:
                power = 0.0
            else:
                hour_of_day = (t * dt_seconds / 3600) % 24
                heating_active = 7 <= hour_of_day < 22

                if heating_active:
                    temp_error = target_temp - temperatures[node_id]
                    power = min(max(temp_error * heating_gain, 0), max_power)
                else:
                    temp_error = 16.0 - temperatures[node_id]
                    power = min(max(temp_error * heating_gain * 0.5, 0), max_power * 0.3)

            heating_powers[node_id] = power
            histories[node_id].heating_power.append(power)

        # Update temperatures
        new_temps: dict[str, float] = {}
        for node_id in graph.node_ids:
            C_i = true_thermal_mass[node_id]
            P_i = heating_powers[node_id]

            # Heat flow from neighbors
            Q_neighbors = 0.0
            for neighbor_id, edge_idx in graph.get_neighbors(node_id):
                edge = graph.edges[edge_idx]
                key = (edge.node_a, edge.node_b)
                G_ij = true_conductance.get(key, true_conductance.get((key[1], key[0]), 0.0))

                if neighbor_id == graph.external_node_id:
                    T_neighbor = ext_temp
                else:
                    T_neighbor = temperatures[neighbor_id]

                Q_neighbors += G_ij * (T_neighbor - temperatures[node_id])

            # Add extra heat loss from window events
            for event in window_events:
                if (
                    event.room_id == node_id
                    and event.start_timestep <= t < event.start_timestep + event.duration_timesteps
                ):
                    Q_neighbors += event.extra_conductance * (ext_temp - temperatures[node_id])

            # Temperature change
            dT = (P_i + Q_neighbors) * dt_seconds / C_i
            dT += rng.normal(0, noise_std)
            new_temps[node_id] = temperatures[node_id] + dT

        temperatures = new_temps

    return histories, external_temps


# -----------------------------------------------------------------------------
# Test Functions
# -----------------------------------------------------------------------------


def test_basic_estimation() -> None:
    """Test basic parameter estimation with no anomalies."""
    print("=" * 60)
    print("Test 1: Basic Parameter Estimation")
    print("=" * 60)

    # Simple 3-room layout:
    #   [Room A] -- [Room B] -- [Room C]
    #      |                       |
    #   exterior                exterior

    room_ids = ["room_a", "room_b", "room_c"]
    adjacency = {
        "room_a": ["room_b"],
        "room_b": ["room_a", "room_c"],
        "room_c": ["room_b"],
    }

    graph = build_graph_from_adjacency(
        room_ids=room_ids,
        adjacency=adjacency,
        exterior_rooms={"room_a", "room_c"},
    )

    # True parameters
    true_mass = {
        "room_a": 300_000.0,
        "room_b": 500_000.0,
        "room_c": 400_000.0,
    }
    true_cond = {
        ("room_a", "room_b"): 15.0,
        ("room_b", "room_c"): 12.0,
        ("room_a", "exterior"): 5.0,
        ("room_c", "exterior"): 6.0,
    }

    print("\nTrue parameters:")
    print("  Thermal mass (kJ/K):", {k: v / 1000 for k, v in true_mass.items()})
    print("  Conductance (W/K):", true_cond)

    # Generate synthetic data
    print("\nGenerating 14 days of data...")
    histories, ext_temps = simulate_building(
        true_thermal_mass=true_mass,
        true_conductance=true_cond,
        graph=graph,
        n_days=14,
        measurements_per_hour=4,
    )

    # Assess data quality
    print("\nAssessing data quality...")
    dt_seconds = 900.0  # 15 min
    quality = assess_building_data_quality(histories, dt_seconds)
    readiness = check_estimation_readiness(quality)
    print(f"  Quality: {readiness.quality_summary}")
    print(f"  Ready: {readiness.ready}")

    # Run estimation
    print("\nRunning estimation...")
    result = estimate_parameters(
        room_data=histories,
        graph=graph,
        external=ExternalConditions(temperature=ext_temps),
        config=EstimationConfig(dt_seconds=dt_seconds, regularization=0.001),
    )

    print("\nResults:")
    print(f"  Success: {result.success}")
    print(f"  R-squared: {result.fit_metrics.r_squared:.4f}")
    print(f"  RMSE: {result.fit_metrics.rmse:.4f} K")

    # Compare estimates to true values
    print("\nEstimated vs True:")
    print(f"  {'Room':<10} {'Est Mass':>12} {'True Mass':>12} {'Error':>10}")
    for room_id in room_ids:
        est = result.parameters.rooms[room_id].thermal_mass / 1000
        true = true_mass[room_id] / 1000
        err = 100 * (est - true) / true
        print(f"  {room_id:<10} {est:>12.1f} {true:>12.1f} {err:>+10.1f}%")


def test_anomaly_detection() -> None:
    """Test anomaly detection with window open events."""
    print("\n" + "=" * 60)
    print("Test 2: Anomaly Detection")
    print("=" * 60)

    room_ids = ["room_a", "room_b"]
    graph = build_graph_from_adjacency(
        room_ids=room_ids,
        adjacency={"room_a": ["room_b"], "room_b": ["room_a"]},
        exterior_rooms={"room_a", "room_b"},
    )

    true_mass = {"room_a": 400_000.0, "room_b": 400_000.0}
    true_cond = {
        ("room_a", "room_b"): 10.0,
        ("room_a", "exterior"): 5.0,
        ("room_b", "exterior"): 5.0,
    }

    # Create window events
    tph = 4  # timesteps per hour
    tpd = 24 * tph  # timesteps per day
    window_events = [
        # Day 3: window open for 2 hours in room_a
        WindowOpenEvent("room_a", 3 * tpd + 10 * tph, 2 * tph, 40.0),
        # Day 5: window open overnight in room_b
        WindowOpenEvent("room_b", 5 * tpd + 22 * tph, 8 * tph, 30.0),
        # Day 8: window open briefly in room_a
        WindowOpenEvent("room_a", 8 * tpd + 14 * tph, 1 * tph, 50.0),
    ]

    print(f"\nSimulating with {len(window_events)} window events...")
    histories, ext_temps = simulate_building(
        true_thermal_mass=true_mass,
        true_conductance=true_cond,
        graph=graph,
        n_days=14,
        window_events=window_events,
    )

    # Estimate with robust loss
    result = estimate_parameters(
        room_data=histories,
        graph=graph,
        external=ExternalConditions(temperature=ext_temps),
        config=EstimationConfig(dt_seconds=900.0, loss="huber"),
    )

    print("\nEstimation with Huber loss:")
    print(f"  R-squared: {result.fit_metrics.r_squared:.4f}")

    # Detect anomalies
    report = detect_anomalies_from_residuals(
        result.residuals,
        AnomalyDetectionConfig(threshold_std=3.0, min_consecutive=2),
    )

    sustained = detect_sustained_events(report.anomalies, min_duration_timesteps=4)
    summary = summarize_anomalies(report)

    print(f"\nDetected {len(report.anomalies)} anomalies, {len(sustained)} sustained:")
    for anomaly in sustained[:5]:
        duration = anomaly.end_timestep - anomaly.start_timestep + 1
        day = anomaly.start_timestep // tpd
        hour = (anomaly.start_timestep % tpd) / tph
        print(f"  {anomaly.room_id}: day {day}, {hour:.0f}:00, duration {duration} steps, type={anomaly.anomaly_type}")

    print("\nSummary by room:")
    for room_id, stats in summary.items():
        print(f"  {room_id}: {stats['n_events']} events, {stats['heat_loss_events']} heat loss")


def test_cooling_curve_analysis() -> None:
    """Test cooling curve analysis for parameter separation."""
    print("\n" + "=" * 60)
    print("Test 3: Cooling Curve Analysis")
    print("=" * 60)

    room_ids = ["room_a"]
    graph = build_graph_from_adjacency(
        room_ids=room_ids,
        adjacency={},
        exterior_rooms={"room_a"},
    )

    # Simple room with only exterior connection
    true_mass = {"room_a": 500_000.0}  # 500 kJ/K
    true_cond = {("room_a", "exterior"): 8.0}  # 8 W/K

    # Expected time constant: τ = R*C = C/G = 500000/8 = 62500 s ≈ 17.4 hours
    expected_tau = true_mass["room_a"] / true_cond[("room_a", "exterior")]
    print(f"\nExpected time constant: {expected_tau:.0f}s = {expected_tau / 3600:.1f}h")

    # Simulate with heating off to observe cooling
    print("\nSimulating cooling (no heating)...")
    histories, ext_temps = simulate_building(
        true_thermal_mass=true_mass,
        true_conductance=true_cond,
        graph=graph,
        n_days=7,
        heating_enabled=False,
        initial_temps={"room_a": 25.0},  # Start warm
    )

    # Find cooling segments
    segments = find_cooling_segments(
        histories["room_a"],
        ext_temps,
        power_threshold=50.0,
        min_duration_timesteps=20,
    )
    print(f"\nFound {len(segments)} cooling segments")

    # Analyze identifiability
    ident = analyze_identifiability(histories["room_a"], ext_temps)
    print("\nIdentifiability analysis:")
    print(f"  Has cooling data: {ident.has_cooling_data}")
    print(f"  Has steady state: {ident.has_steady_state_data}")
    print(f"  Can separate params: {ident.can_separate_parameters}")
    print(f"  Recommendation: {ident.recommended_approach}")


def test_heat_flow_mapping() -> None:
    """Test heat flow computation and thermal bridge detection."""
    print("\n" + "=" * 60)
    print("Test 4: Heat Flow Mapping")
    print("=" * 60)

    room_ids = ["room_a", "room_b", "room_c"]
    graph = build_graph_from_adjacency(
        room_ids=room_ids,
        adjacency={"room_a": ["room_b"], "room_b": ["room_a", "room_c"], "room_c": ["room_b"]},
        exterior_rooms={"room_a", "room_c"},
    )

    true_mass = {"room_a": 300_000.0, "room_b": 500_000.0, "room_c": 400_000.0}
    # room_c has poor insulation (high exterior conductance)
    true_cond = {
        ("room_a", "room_b"): 12.0,
        ("room_b", "room_c"): 15.0,
        ("room_a", "exterior"): 5.0,
        ("room_c", "exterior"): 15.0,  # Thermal bridge!
    }

    histories, ext_temps = simulate_building(
        true_thermal_mass=true_mass,
        true_conductance=true_cond,
        graph=graph,
        n_days=14,
    )

    result = estimate_parameters(
        room_data=histories,
        graph=graph,
        external=ExternalConditions(temperature=ext_temps),
    )

    print(f"\nEstimation R-squared: {result.fit_metrics.r_squared:.4f}")

    # Compute heat flow snapshot
    room_temps = {rid: histories[rid].temperature[100] for rid in room_ids}
    snapshot = compute_heat_flow_snapshot(
        graph=graph,
        parameters=result.parameters,
        room_temps=room_temps,
        external_temp=ext_temps[100],
    )

    print("\nHeat flow at timestep 100:")
    for flow in snapshot.flows:
        print(f"  {flow.from_room} -> {flow.to_room}: {flow.flow_watts:.1f} W")

    print("\nNet heat flow by room:")
    for room_id, net in snapshot.net_by_room.items():
        print(f"  {room_id}: {net:+.1f} W")

    # Identify thermal bridges
    bridges = identify_thermal_bridges(result.parameters, graph)
    print("\nThermal bridges detected:")
    for bridge in bridges:
        print(f"  {bridge.room_ids}: G={bridge.conductance:.1f} W/K, severity={bridge.severity}")


def main() -> None:
    """Run all tests."""
    test_basic_estimation()
    test_anomaly_detection()
    test_cooling_curve_analysis()
    test_heat_flow_mapping()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
