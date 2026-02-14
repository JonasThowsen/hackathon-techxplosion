"""Realistic stress test for thermal estimation.

This test simulates conditions closer to real buildings:
- 12-room office floor layout
- Unmodeled disturbances (solar, occupancy, equipment)
- Realistic sensor noise (±0.3°C)
- Time-varying infiltration
- Model mismatch (non-linear effects we don't model)

Expected results: R² of 0.7-0.85, parameter errors of 20-50%
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from analysis.thermal import (
    AnomalyDetectionConfig,
    EstimationConfig,
    ExternalConditions,
    RoomTimeSeries,
    ThermalGraph,
    assess_building_data_quality,
    build_graph_from_adjacency,
    check_estimation_readiness,
    detect_anomalies_from_residuals,
    detect_sustained_events,
    estimate_parameters,
    identify_thermal_bridges,
)

# -----------------------------------------------------------------------------
# Realistic Building Layout
# -----------------------------------------------------------------------------

# 12-room office floor:
#
#   [Office1] [Office2] [Office3] [Conference]
#       |         |         |          |
#   [Corridor-----------------------------]
#       |         |         |          |
#   [Office4] [Office5] [Kitchen] [Server]
#       |                              |
#   exterior                       exterior
#
# - Offices: standard thermal mass
# - Conference: large room, more glass (solar gains)
# - Kitchen: equipment heat loads
# - Server room: constant high heat load
# - Corridor: connects everything, exterior doors

ROOM_IDS = [
    "office1",
    "office2",
    "office3",
    "conference",
    "corridor",
    "office4",
    "office5",
    "kitchen",
    "server",
]

ADJACENCY = {
    "office1": ["office2", "corridor"],
    "office2": ["office1", "office3", "corridor"],
    "office3": ["office2", "conference", "corridor"],
    "conference": ["office3", "corridor"],
    "corridor": ["office1", "office2", "office3", "conference", "office4", "office5", "kitchen", "server"],
    "office4": ["office5", "corridor"],
    "office5": ["office4", "kitchen", "corridor"],
    "kitchen": ["office5", "server", "corridor"],
    "server": ["kitchen", "corridor"],
}

# Rooms with exterior walls
EXTERIOR_ROOMS = {"office1", "office4", "conference", "server", "corridor"}

# True thermal parameters (what we're trying to estimate)
TRUE_THERMAL_MASS = {
    "office1": 400_000.0,  # 400 kJ/K - small office
    "office2": 450_000.0,  # 450 kJ/K - interior office
    "office3": 400_000.0,  # 400 kJ/K - small office
    "conference": 800_000.0,  # 800 kJ/K - large room
    "corridor": 600_000.0,  # 600 kJ/K - long corridor
    "office4": 400_000.0,  # 400 kJ/K - small office
    "office5": 450_000.0,  # 450 kJ/K - interior office
    "kitchen": 350_000.0,  # 350 kJ/K - smaller room
    "server": 300_000.0,  # 300 kJ/K - small room, equipment
}

TRUE_CONDUCTANCE = {
    # Interior walls (well insulated)
    ("office1", "office2"): 12.0,
    ("office2", "office3"): 12.0,
    ("office3", "conference"): 15.0,  # Larger shared wall
    ("office4", "office5"): 12.0,
    ("office5", "kitchen"): 10.0,
    ("kitchen", "server"): 8.0,
    # Corridor connections (doors, so higher conductance)
    ("office1", "corridor"): 18.0,
    ("office2", "corridor"): 18.0,
    ("office3", "corridor"): 18.0,
    ("conference", "corridor"): 25.0,  # Big doors
    ("office4", "corridor"): 18.0,
    ("office5", "corridor"): 18.0,
    ("kitchen", "corridor"): 20.0,
    ("server", "corridor"): 15.0,
    # Exterior walls
    ("office1", "exterior"): 6.0,
    ("office4", "exterior"): 6.0,
    ("conference", "exterior"): 12.0,  # Large windows!
    ("server", "exterior"): 4.0,  # Well insulated
    ("corridor", "exterior"): 8.0,  # Entrance doors
}


# -----------------------------------------------------------------------------
# Disturbance Models (Unmodeled by estimator)
# -----------------------------------------------------------------------------


@dataclass
class OccupancyPattern:
    """Occupancy heat load pattern for a room."""

    room_id: str
    weekday_schedule: dict[int, float]  # hour -> heat load (W)
    weekend_factor: float = 0.1  # Fraction of weekday load


def get_occupancy_heat(
    room_id: str,
    patterns: dict[str, OccupancyPattern],
    hour: float,
    is_weekend: bool,
    rng: np.random.Generator,
) -> float:
    """Get occupancy heat load with randomness."""
    if room_id not in patterns:
        return 0.0

    pattern = patterns[room_id]
    hour_int = int(hour) % 24
    base_load = pattern.weekday_schedule.get(hour_int, 0.0)

    if is_weekend:
        base_load *= pattern.weekend_factor

    # Add randomness (people come and go)
    noise = rng.normal(0, base_load * 0.3) if base_load > 0 else 0
    return max(0, base_load + noise)


def get_solar_gain_unmodeled(
    room_id: str,
    hour: float,
    day_of_year: int,
    cloud_cover: float,
    rng: np.random.Generator,
) -> float:
    """Unmodeled solar gain - realistic winter values.

    In winter with low sun angle, solar gains are modest.
    Target: peak ~200W for south-facing room, ~50-100W typical.
    """
    _ = day_of_year  # Fixed to winter for this test

    solar_rooms = {
        "conference": 1.0,
        "office1": 0.4,
        "office4": 0.4,
        "corridor": 0.2,
    }

    if room_id not in solar_rooms:
        return 0.0

    exposure = solar_rooms[room_id]

    # Winter daylight hours
    if hour < 8 or hour > 16:
        return 0.0

    # Solar profile
    solar_intensity = max(0, np.sin(np.pi * (hour - 8) / 8))

    # Cloud effect
    clear_factor = 1.0 - 0.6 * cloud_cover

    # Base: 200W peak for conference room on clear winter day
    base_gain = 200.0

    gain = base_gain * exposure * solar_intensity * clear_factor
    gain += rng.normal(0, gain * 0.1) if gain > 0 else 0

    return max(0, gain)


def get_equipment_heat(
    room_id: str,
    hour: float,
    is_weekend: bool,
    rng: np.random.Generator,
) -> float:
    """Equipment heat loads - realistic scale relative to HVAC.

    Target: ~10-20% of typical heating power as unmodeled disturbance.
    Typical heating is 500-2000W, so equipment should be 50-300W range.
    """
    # Server room: small network closet, not a data center
    if room_id == "server":
        return 200.0 + rng.normal(0, 20)  # 200W constant

    # Kitchen: minor loads
    if room_id == "kitchen":
        if is_weekend:
            return 0.0
        if 12 <= hour < 13:  # Lunch only
            return 150 + rng.normal(0, 30)
        return 20 + rng.uniform(0, 30)

    # Offices: 1 computer
    if room_id.startswith("office"):
        if is_weekend:
            return 0.0
        if 9 <= hour < 17:
            return 80 + rng.normal(0, 20)
        return 0.0

    # Conference: occasional
    if room_id == "conference":
        if is_weekend:
            return 0.0
        if 10 <= hour < 16 and rng.random() < 0.2:
            return 150 + rng.normal(0, 30)
        return 0.0

    return 0.0


def get_infiltration_factor(
    hour: float,
    external_temp: float,
    wind_speed: float,
) -> float:
    """Time-varying infiltration multiplier.

    Infiltration increases with:
    - Wind speed
    - Temperature difference (stack effect)
    - Door usage (work hours)
    """
    # Base infiltration
    base = 1.0

    # Wind effect
    wind_factor = 1.0 + 0.1 * wind_speed  # wind_speed in m/s

    # Stack effect (more infiltration with larger ΔT)
    stack_factor = 1.0 + 0.02 * abs(20 - external_temp)

    # Door usage during work hours
    if 8 <= hour < 18:
        door_factor = 1.3
    else:
        door_factor = 1.0

    return base * wind_factor * stack_factor * door_factor


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------


def generate_weather(
    n_timesteps: int,
    dt_seconds: float,
    base_temp: float = 5.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate realistic weather data.

    Returns:
        Tuple of (external_temps, wind_speeds, cloud_cover)
    """
    rng = np.random.default_rng(seed)

    external_temps = np.zeros(n_timesteps)
    wind_speeds = np.zeros(n_timesteps)
    cloud_cover = np.zeros(n_timesteps)

    for t in range(n_timesteps):
        hours = (t * dt_seconds) / 3600
        hour_of_day = hours % 24
        day = int(hours // 24)

        # Daily temperature cycle
        daily_var = 4.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)

        # Day-to-day variation (weather fronts)
        day_rng = np.random.default_rng(seed + day)
        day_offset = day_rng.uniform(-5, 5)

        external_temps[t] = base_temp + daily_var + day_offset

        # Wind: varies through day, gusts
        wind_speeds[t] = 3.0 + 2.0 * np.sin(2 * np.pi * hour_of_day / 24) + rng.exponential(1.0)

        # Cloud cover: persistent but changes
        if t == 0:
            cloud_cover[t] = rng.uniform(0, 1)
        else:
            # Random walk with mean reversion
            cloud_cover[t] = cloud_cover[t - 1] + rng.normal(0, 0.05)
            cloud_cover[t] = 0.5 + 0.8 * (cloud_cover[t] - 0.5)  # Mean reversion
            cloud_cover[t] = np.clip(cloud_cover[t], 0, 1)

    return external_temps, wind_speeds, cloud_cover


def simulate_realistic_building(
    n_days: int = 30,
    measurements_per_hour: int = 4,
    sensor_noise_std: float = 0.3,  # Realistic sensor noise
    seed: int = 42,
) -> tuple[dict[str, RoomTimeSeries], list[float], ThermalGraph]:
    """Simulate a realistic building with all disturbances.

    Returns:
        Tuple of (room_histories, external_temps, graph)
    """
    rng = np.random.default_rng(seed)

    dt_seconds = 3600.0 / measurements_per_hour
    n_timesteps = n_days * 24 * measurements_per_hour

    # Build graph
    graph = build_graph_from_adjacency(
        room_ids=ROOM_IDS,
        adjacency=ADJACENCY,
        exterior_rooms=EXTERIOR_ROOMS,
    )

    # Generate weather
    external_temps, wind_speeds, cloud_cover = generate_weather(n_timesteps, dt_seconds, base_temp=5.0, seed=seed)

    # Define occupancy patterns - ~80W per person (metabolic heat)
    # Single-occupancy offices, occasional meetings
    work_hours = dict.fromkeys(range(9, 17), 80.0)  # 1 person
    occupancy_patterns = {
        "office1": OccupancyPattern("office1", work_hours),
        "office2": OccupancyPattern("office2", work_hours),
        "office3": OccupancyPattern("office3", work_hours),
        "office4": OccupancyPattern("office4", work_hours),
        "office5": OccupancyPattern("office5", work_hours),
        "conference": OccupancyPattern("conference", dict.fromkeys(range(10, 16), 160.0)),  # 2 people avg
        "kitchen": OccupancyPattern("kitchen", {12: 80.0}),  # Lunch only
        "corridor": OccupancyPattern("corridor", dict.fromkeys(range(9, 17), 20.0)),  # Passing through
    }

    # Initialize temperatures
    temperatures = {room_id: 18.0 + rng.uniform(-2, 2) for room_id in ROOM_IDS}

    # Heating parameters
    target_temp = 21.0
    heating_gain = 400.0
    max_power = 3000.0

    # Initialize histories
    histories: dict[str, RoomTimeSeries] = {
        room_id: RoomTimeSeries(
            room_id=room_id,
            timestamps=[],
            temperature=[],
            heating_power=[],
        )
        for room_id in ROOM_IDS
    }

    start_time = datetime(2024, 1, 15, 0, 0, 0)  # Mid-January

    print(f"Simulating {n_days} days ({n_timesteps} timesteps)...")

    for t in range(n_timesteps):
        timestamp = start_time + timedelta(seconds=t * dt_seconds)
        hours = (t * dt_seconds) / 3600
        hour_of_day = hours % 24
        day_of_week = (int(hours // 24) + 1) % 7  # 0=Mon, 6=Sun
        is_weekend = day_of_week >= 5
        day_of_year = 15 + int(hours // 24)  # Starting Jan 15

        ext_temp = float(external_temps[t])
        wind = float(wind_speeds[t])
        clouds = float(cloud_cover[t])

        # Record current state (with sensor noise!)
        for room_id in ROOM_IDS:
            measured_temp = temperatures[room_id] + rng.normal(0, sensor_noise_std)
            histories[room_id].timestamps.append(timestamp)
            histories[room_id].temperature.append(measured_temp)

        # Compute infiltration factor
        infil_factor = get_infiltration_factor(hour_of_day, ext_temp, wind)

        # Calculate heating power and disturbances
        heating_powers: dict[str, float] = {}
        disturbances: dict[str, float] = {}

        for room_id in ROOM_IDS:
            # Heating control (simple proportional)
            if is_weekend and room_id != "server":
                setpoint = 16.0  # Weekend setback
            elif 7 <= hour_of_day < 20:
                setpoint = target_temp
            else:
                setpoint = 18.0  # Night setback

            temp_error = setpoint - temperatures[room_id]
            power = np.clip(temp_error * heating_gain, 0, max_power)
            heating_powers[room_id] = power
            histories[room_id].heating_power.append(power)

            # Compute unmodeled disturbances
            Q_occupancy = get_occupancy_heat(room_id, occupancy_patterns, hour_of_day, is_weekend, rng)
            Q_solar = get_solar_gain_unmodeled(room_id, hour_of_day, day_of_year, clouds, rng)
            Q_equipment = get_equipment_heat(room_id, hour_of_day, is_weekend, rng)

            disturbances[room_id] = Q_occupancy + Q_solar + Q_equipment

        # Update temperatures
        new_temps: dict[str, float] = {}
        for room_id in ROOM_IDS:
            C_i = TRUE_THERMAL_MASS[room_id]
            P_i = heating_powers[room_id]
            Q_disturb = disturbances[room_id]

            # Heat flow from neighbors (with infiltration effect on exterior)
            Q_neighbors = 0.0
            for neighbor_id, edge_idx in graph.get_neighbors(room_id):
                edge = graph.edges[edge_idx]
                key = (edge.node_a, edge.node_b)
                G_ij = TRUE_CONDUCTANCE.get(key, TRUE_CONDUCTANCE.get((key[1], key[0]), 0.0))

                # Infiltration increases effective conductance for exterior
                if neighbor_id == graph.external_node_id:
                    G_ij *= infil_factor
                    T_neighbor = ext_temp
                else:
                    T_neighbor = temperatures[neighbor_id]

                Q_neighbors += G_ij * (T_neighbor - temperatures[room_id])

            # Temperature change
            dT = (P_i + Q_neighbors + Q_disturb) * dt_seconds / C_i

            # Small non-linearity (model mismatch)
            mass_factor = 1.0 + 0.002 * (temperatures[room_id] - 20)
            dT /= mass_factor

            new_temps[room_id] = temperatures[room_id] + dT

        temperatures = new_temps

    return histories, list(external_temps), graph


# -----------------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------------


def run_realistic_test() -> None:
    """Run estimation on realistic simulated data."""
    print("=" * 70)
    print("REALISTIC STRESS TEST")
    print("=" * 70)
    print()
    print("Simulating 12-room office floor with:")
    print("  - Unmodeled solar gains (up to 2kW in conference room)")
    print("  - Occupancy heat loads")
    print("  - Equipment (3kW server room, kitchen appliances)")
    print("  - Time-varying infiltration")
    print("  - Sensor noise: ±0.3°C")
    print("  - Non-linear thermal mass effects")
    print()

    # Simulate
    histories, ext_temps, graph = simulate_realistic_building(
        n_days=30,
        measurements_per_hour=4,
        sensor_noise_std=0.3,
    )

    # Show data summary
    print("\nData summary:")
    for room_id in ROOM_IDS[:3]:  # Just show first 3
        hist = histories[room_id]
        temps = hist.temperature
        powers = hist.heating_power
        print(f"  {room_id}: T=[{min(temps):.1f}, {max(temps):.1f}]°C, P=[{min(powers):.0f}, {max(powers):.0f}]W")
    print("  ...")

    # Assess data quality
    print("\nData quality assessment:")
    dt_seconds = 900.0
    quality = assess_building_data_quality(histories, dt_seconds)
    readiness = check_estimation_readiness(quality)
    print(f"  {readiness.quality_summary}")

    for room_id, metrics in list(quality.items())[:3]:
        print(
            f"  {room_id}: {metrics.quality_level.value}, "
            f"T_range={metrics.temp_range:.1f}°C, "
            f"cycles={metrics.n_heating_cycles}"
        )

    # Run estimation
    print("\nRunning estimation...")
    result = estimate_parameters(
        room_data=histories,
        graph=graph,
        external=ExternalConditions(temperature=ext_temps),
        config=EstimationConfig(
            dt_seconds=dt_seconds,
            regularization=0.01,
            loss="huber",  # Robust to outliers
        ),
    )

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Success: {result.success}")
    print(f"  R-squared: {result.fit_metrics.r_squared:.4f}")
    print(f"  RMSE: {result.fit_metrics.rmse:.4f} K")
    print(f"  Observations: {result.fit_metrics.n_observations}")

    if result.warnings:
        print(f"\n  Warnings: {len(result.warnings)}")
        for w in result.warnings[:3]:
            print(f"    {w.room_id}: {w.issue}")

    # Compare to true values
    print("\nThermal mass estimation:")
    print(f"  {'Room':<12} {'Estimated':>10} {'True':>10} {'Error':>10}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")

    errors: list[float] = []
    for room_id in ROOM_IDS:
        est = result.parameters.rooms[room_id].thermal_mass / 1000
        true = TRUE_THERMAL_MASS[room_id] / 1000
        err = 100 * (est - true) / true
        errors.append(abs(err))
        print(f"  {room_id:<12} {est:>10.0f} {true:>10.0f} {err:>+10.1f}%")

    print(f"\n  Mean absolute error: {np.mean(errors):.1f}%")
    print(f"  Max absolute error: {np.max(errors):.1f}%")

    # Conductance estimation
    print("\nConductance estimation (sample):")
    print(f"  {'Edge':<25} {'Est':>8} {'True':>8} {'Error':>8}")
    sample_edges = [
        ("office1", "office2"),
        ("conference", "corridor"),
        ("conference", "exterior"),
        ("server", "exterior"),
    ]
    for edge in sample_edges:
        est = result.parameters.get_conductance(edge[0], edge[1])
        true = TRUE_CONDUCTANCE.get(edge, TRUE_CONDUCTANCE.get((edge[1], edge[0]), 0))
        if true > 0:
            err = 100 * (est - true) / true
            print(f"  {edge[0]}-{edge[1]:<14} {est:>8.1f} {true:>8.1f} {err:>+8.1f}%")

    # Anomaly detection
    print("\nAnomaly detection:")
    report = detect_anomalies_from_residuals(
        result.residuals,
        AnomalyDetectionConfig(threshold_std=2.5, min_consecutive=2),
    )
    sustained = detect_sustained_events(report.anomalies, min_duration_timesteps=4)
    print(f"  Total anomalies: {len(report.anomalies)}")
    print(f"  Sustained events: {len(sustained)}")

    # Rooms with most anomalies
    anomaly_counts = {room_id: len(ts) for room_id, ts in report.anomalous_timesteps_by_room.items()}
    top_anomaly_rooms = sorted(anomaly_counts.items(), key=lambda x: -x[1])[:3]
    print("  Rooms with most anomalies:")
    for room_id, count in top_anomaly_rooms:
        print(f"    {room_id}: {count} anomalous timesteps")

    # Thermal bridges
    print("\nThermal bridges (poorly insulated connections):")
    bridges = identify_thermal_bridges(result.parameters, graph)
    for bridge in bridges[:5]:
        print(f"  {bridge.room_ids}: G={bridge.conductance:.1f} W/K, severity={bridge.severity}")

    print("\n" + "=" * 70)
    print("Expected: R² ~ 0.7-0.85, errors ~ 20-50%")
    print("=" * 70)


if __name__ == "__main__":
    run_realistic_test()
