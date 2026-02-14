"""Single-room first-order RC thermal model.

This module implements the discrete-time RC model for a single thermal zone.
The governing equation is:

    C * dT/dt = P + Σ G_j * (T_j - T) + G_ext * (T_ext - T)

In discrete form with timestep dt:

    T(t+1) = T(t) + (dt/C) * [P(t) + Σ G_j * (T_j(t) - T(t))]

This can be rearranged into linear regression form for parameter estimation.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from analysis.thermal.types import ThermalGraph, ThermalParameters


def predict_temperature_change(
    thermal_mass: float,
    heating_power: float,
    current_temp: float,
    neighbor_temps: list[float],
    neighbor_conductances: list[float],
    dt_seconds: float,
) -> float:
    """Predict temperature change for a single timestep.

    This implements the discrete governing equation:
        dT = (dt/C) * [P + Σ G_j * (T_j - T)]

    Args:
        thermal_mass: Room thermal capacitance C (J/K)
        heating_power: HVAC power input P (W)
        current_temp: Current room temperature T (°C)
        neighbor_temps: Temperatures of neighboring zones (°C)
        neighbor_conductances: Conductances to each neighbor (W/K)
        dt_seconds: Time step duration (seconds)

    Returns:
        Predicted temperature change dT (K or °C, same units as input)
    """
    # Heat input from heating
    q_heating = heating_power

    # Heat flow from neighbors: Q = Σ G_j * (T_j - T)
    q_neighbors = sum(
        g * (t_neighbor - current_temp) for g, t_neighbor in zip(neighbor_conductances, neighbor_temps, strict=True)
    )

    # Total heat flow into the room
    q_total = q_heating + q_neighbors

    # Temperature change: dT = Q * dt / C
    return q_total * dt_seconds / thermal_mass


def predict_temperature_trajectory(
    thermal_mass: float,
    heating_powers: NDArray[np.float64],
    initial_temp: float,
    neighbor_temp_series: list[NDArray[np.float64]],
    neighbor_conductances: list[float],
    dt_seconds: float,
) -> NDArray[np.float64]:
    """Predict temperature trajectory over multiple timesteps.

    Args:
        thermal_mass: Room thermal capacitance C (J/K)
        heating_powers: Power input time series (W), shape (n_timesteps,)
        initial_temp: Initial temperature T_0 (°C)
        neighbor_temp_series: Temperature series for each neighbor, list of (n_timesteps,) arrays
        neighbor_conductances: Conductance to each neighbor (W/K)
        dt_seconds: Time step duration (seconds)

    Returns:
        Predicted temperatures, shape (n_timesteps,)
    """
    n_timesteps = len(heating_powers)
    temperatures = np.zeros(n_timesteps, dtype=np.float64)
    temperatures[0] = initial_temp

    for t in range(n_timesteps - 1):
        neighbor_temps = [series[t] for series in neighbor_temp_series]
        dt = predict_temperature_change(
            thermal_mass=thermal_mass,
            heating_power=float(heating_powers[t]),
            current_temp=float(temperatures[t]),
            neighbor_temps=neighbor_temps,
            neighbor_conductances=neighbor_conductances,
            dt_seconds=dt_seconds,
        )
        temperatures[t + 1] = temperatures[t] + dt

    return temperatures


def compute_steady_state_temperature(
    heating_power: float,
    external_temp: float,
    exterior_conductance: float,
    neighbor_temps: list[float] | None = None,
    neighbor_conductances: list[float] | None = None,
) -> float:
    """Compute steady-state temperature for a room.

    At steady state, dT/dt = 0, so:
        P + Σ G_j * (T_j - T) + G_ext * (T_ext - T) = 0

    Solving for T:
        T = (P + Σ G_j*T_j + G_ext*T_ext) / (Σ G_j + G_ext)

    Args:
        heating_power: HVAC power input P (W)
        external_temp: External temperature T_ext (°C)
        exterior_conductance: Conductance to outside G_ext (W/K)
        neighbor_temps: Temperatures of neighboring rooms (°C)
        neighbor_conductances: Conductances to neighbors (W/K)

    Returns:
        Steady-state temperature (°C)
    """
    neighbor_temps = neighbor_temps or []
    neighbor_conductances = neighbor_conductances or []

    # Total conductance
    total_conductance = exterior_conductance + sum(neighbor_conductances)

    if total_conductance == 0:
        raise ValueError("Cannot compute steady state: total conductance is zero")

    # Weighted temperature sum
    weighted_sum = heating_power + exterior_conductance * external_temp
    for t_n, g_n in zip(neighbor_temps, neighbor_conductances, strict=True):
        weighted_sum += g_n * t_n

    return weighted_sum / total_conductance


def estimate_resistance_from_steady_state(
    indoor_temp: float,
    external_temp: float,
    heating_power: float,
) -> float:
    """Estimate effective thermal resistance from steady-state conditions.

    At steady state with only external heat loss:
        P = (T_indoor - T_ext) / R

    So:
        R = (T_indoor - T_ext) / P

    Args:
        indoor_temp: Indoor temperature (°C)
        external_temp: External temperature (°C)
        heating_power: Heating power maintaining the temperature (W)

    Returns:
        Effective thermal resistance R (K/W)

    Raises:
        ValueError: If heating power is zero
    """
    if heating_power == 0:
        raise ValueError("Cannot estimate resistance: heating power is zero")

    return (indoor_temp - external_temp) / heating_power


def compute_time_constant(thermal_mass: float, thermal_resistance: float) -> float:
    """Compute the RC time constant.

    The time constant τ = R * C determines how fast the room
    responds to temperature changes.

    Args:
        thermal_mass: Thermal capacitance C (J/K)
        thermal_resistance: Thermal resistance R (K/W)

    Returns:
        Time constant τ (seconds)
    """
    return thermal_mass * thermal_resistance


def build_regression_matrices(
    temperatures: NDArray[np.float64],
    heating_powers: NDArray[np.float64],
    neighbor_temps_list: list[NDArray[np.float64]],
    external_temps: NDArray[np.float64],
    dt_seconds: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build regression matrices for linear parameter estimation.

    The discrete model can be written as:
        T(t+1) = a*T(t) + Σ b_j*T_j(t) + c*T_ext(t) + d*P(t)

    Where the coefficients depend on C and the G values.

    For a single room with only external connection:
        T(t+1) - T(t) = (dt/C) * [P(t) + G_ext*(T_ext(t) - T(t))]
        dT(t) = (dt*P(t))/C + (dt*G_ext/C)*T_ext(t) - (dt*G_ext/C)*T(t)

    Let θ = [1/C, G_ext/C]^T
    Then: dT(t) = [dt*P(t), dt*(T_ext(t) - T(t))] @ θ

    This returns X and y such that y ≈ X @ θ in a least squares sense.

    Args:
        temperatures: Room temperature time series (n_timesteps,)
        heating_powers: Heating power time series (n_timesteps,)
        neighbor_temps_list: List of neighbor temperature series
        external_temps: External temperature series (n_timesteps,)
        dt_seconds: Time step duration

    Returns:
        Tuple of (X, y) where X is (n_samples, n_features) and y is (n_samples,)
    """
    n = len(temperatures) - 1  # Number of transitions
    n_neighbors = len(neighbor_temps_list)

    # Features: [dt*P, dt*(T_ext - T), dt*(T_neighbor_1 - T), ...]
    n_features = 1 + 1 + n_neighbors  # power term, external, neighbors

    X = np.zeros((n, n_features), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)

    for t in range(n):
        # Output: observed temperature change
        y[t] = temperatures[t + 1] - temperatures[t]

        # Features
        col = 0

        # Power term: contributes (dt/C) * P
        X[t, col] = dt_seconds * heating_powers[t]
        col += 1

        # External temperature term: contributes (dt*G_ext/C) * (T_ext - T)
        X[t, col] = dt_seconds * (external_temps[t] - temperatures[t])
        col += 1

        # Neighbor terms: each contributes (dt*G_j/C) * (T_j - T)
        for neighbor_temps in neighbor_temps_list:
            X[t, col] = dt_seconds * (neighbor_temps[t] - temperatures[t])
            col += 1

    return X, y


# -----------------------------------------------------------------------------
# Building-Level Prediction
# -----------------------------------------------------------------------------


@dataclass
class BuildingPrediction:
    """Result of building-wide temperature prediction."""

    room_temps: dict[str, NDArray[np.float64]]
    heating_powers: dict[str, NDArray[np.float64]]
    external_temps: NDArray[np.float64]


def predict_building_temperature(
    parameters: ThermalParameters,
    graph: ThermalGraph,
    initial_temps: dict[str, float],
    heating_schedule: dict[str, NDArray[np.float64]],
    external_temps: NDArray[np.float64],
    dt_seconds: float,
) -> BuildingPrediction:
    """Predict building-wide temperature trajectory over multiple timesteps.

    This is the main entry point for forward prediction. Given estimated
    thermal parameters and a future heating schedule, it predicts how
    temperatures will evolve across all rooms.

    This is useful for:
    - MPC-based power optimization
    - What-if scenario analysis
    - Predictive control

    Args:
        parameters: Estimated thermal parameters from estimation
        graph: Thermal network topology
        initial_temps: Current temperature of each room (°C)
        heating_schedule: Future heating power for each room (W), shape (n_timesteps,)
            Each entry should have the same length
        external_temps: Future external temperatures (°C), shape (n_timesteps,)
        dt_seconds: Time step duration (seconds)

    Returns:
        BuildingPrediction with predicted temperatures per room
    """
    if not heating_schedule:
        raise ValueError("heating_schedule cannot be empty")

    n_timesteps = len(next(iter(heating_schedule.values())))

    if len(external_temps) != n_timesteps:
        raise ValueError(
            f"external_temps length ({len(external_temps)}) must match heating_schedule length ({n_timesteps})"
        )

    for room_id, powers in heating_schedule.items():
        if len(powers) != n_timesteps:
            raise ValueError(
                f"heating_schedule[{room_id}] length ({len(powers)}) does not match expected ({n_timesteps})"
            )

    temps: dict[str, NDArray[np.float64]] = {
        room_id: np.zeros(n_timesteps, dtype=np.float64) for room_id in graph.node_ids
    }
    for room_id in graph.node_ids:
        temps[room_id][0] = initial_temps.get(room_id, 20.0)

    for t in range(n_timesteps - 1):
        for room_id in graph.node_ids:
            room_params = parameters.rooms[room_id]
            C = room_params.thermal_mass
            P = heating_schedule[room_id][t]
            T_current = temps[room_id][t]

            q_neighbors = 0.0
            for neighbor_id, edge_idx in graph.get_neighbors(room_id):
                if neighbor_id == graph.external_node_id:
                    T_neighbor = external_temps[t]
                    g = room_params.exterior_conductance or 0.0
                else:
                    T_neighbor = temps[neighbor_id][t]
                    edge = graph.edges[edge_idx]
                    key = tuple(sorted([edge.node_a, edge.node_b]))
                    g = parameters.get_conductance(key[0], key[1])

                q_neighbors += g * (T_neighbor - T_current)

            dT = (P + q_neighbors) * dt_seconds / C
            temps[room_id][t + 1] = temps[room_id][t] + dT

    return BuildingPrediction(
        room_temps=temps,
        heating_powers=heating_schedule,
        external_temps=external_temps,
    )
