"""Single-room first-order RC thermal model.

This module implements the discrete-time RC model for a single thermal zone.
The governing equation is:

    C * dT/dt = P + Σ G_j * (T_j - T) + G_ext * (T_ext - T)

In discrete form with timestep dt:

    T(t+1) = T(t) + (dt/C) * [P(t) + Σ G_j * (T_j(t) - T(t))]

This can be rearranged into linear regression form for parameter estimation.

Also provides first-principles parameter estimation from geometry.
"""

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from analysis.thermal.types import (
    RoomParameters,
    ThermalGraph,
    ThermalParameters,
)


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


# -----------------------------------------------------------------------------
# First-Principles Parameter Estimation
# -----------------------------------------------------------------------------


@dataclass
class UValuesNorway:
    """U-values (W/m²·K) for Norwegian public buildings."""

    facade_incl_windows: float = 0.40
    opaque_wall_only: float = 0.17
    window_only: float = 0.90
    roof: float = 0.13
    internal_wall: float = 2.0
    internal_to_unheated: float = 0.35

    def for_wall(self, wall_type: str) -> float:
        """Get U-value for wall type."""
        mapping = {
            "exterior": self.facade_incl_windows,
            "opaque": self.opaque_wall_only,
            "window": self.window_only,
            "roof": self.roof,
            "internal": self.internal_wall,
            "unheated": self.internal_to_unheated,
        }
        return mapping.get(wall_type, self.facade_incl_windows)


class RoomGeometry:
    """Room geometry for first-principles calculations."""

    room_id: str
    floor_area_m2: float
    wall_area_m2: float
    window_area_m2: float
    ceiling_height_m: float
    is_exterior: bool
    neighbor_ids: list[str]

    def __init__(
        self,
        room_id: str,
        floor_area_m2: float,
        wall_area_m2: float,
        window_area_m2: float = 0.0,
        ceiling_height_m: float = 3.0,
        is_exterior: bool = True,
        neighbor_ids: list[str] | None = None,
    ) -> None:
        self.room_id = room_id
        self.floor_area_m2 = floor_area_m2
        self.wall_area_m2 = wall_area_m2
        self.window_area_m2 = window_area_m2
        self.ceiling_height_m = ceiling_height_m
        self.is_exterior = is_exterior
        self.neighbor_ids = neighbor_ids or []


def estimate_thermal_mass(
    floor_area_m2: float,
    ceiling_height_m: float = 3.0,
    occupancy: str = "medium",
) -> float:
    """Estimate thermal mass from room geometry.

    Based on typical heat capacity of:
    - Air: ~1.2 kJ/m³·K
    - Furniture/contents: ~50-100 kJ/m² floor area
    - Building structure: depends on construction

    Args:
        floor_area_m2: Floor area in m²
        ceiling_height_m: Ceiling height in meters
        occupancy: "light", "medium", or "heavy" (furnishings)

    Returns:
        Thermal mass in J/K
    """
    volume = floor_area_m2 * ceiling_height_m

    air_mass_j_k = volume * 1200.0

    furnishing_mass = {
        "light": 20_000,
        "medium": 50_000,
        "heavy": 100_000,
    }
    furnishing_mass_j_k = floor_area_m2 * furnishing_mass[occupancy]

    structure_mass_j_k = floor_area_m2 * ceiling_height_m * 500.0

    return air_mass_j_k + furnishing_mass_j_k + structure_mass_j_k


def estimate_conductance(
    area_m2: float,
    u_value: float,
) -> float:
    """Estimate thermal conductance from area and U-value.

    G = U * A (W/K)

    Args:
        area_m2: Wall/area in m²
        u_value: U-value in W/m²·K

    Returns:
        Conductance in W/K
    """
    return area_m2 * u_value


def compute_conductance_from_rooms(
    rooms: list[RoomGeometry],
    u_values: UValuesNorway | None = None,
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Compute conductance parameters from room geometry.

    Args:
        rooms: List of room geometries
        u_values: U-values to use (defaults to Norwegian standards)

    Returns:
        Tuple of (room_thermal_mass dict, conductance dict)
    """
    if u_values is None:
        u_values = UValuesNorway()

    room_masses: dict[str, float] = {}
    conductances: dict[tuple[str, str], float] = {}

    room_by_id = {r.room_id: r for r in rooms}

    for room in rooms:
        mass = estimate_thermal_mass(room.floor_area_m2, room.ceiling_height_m)
        room_masses[room.room_id] = mass

        if room.is_exterior:
            opaque_area = room.wall_area_m2 - room.window_area_m2
            g_wall = estimate_conductance(opaque_area, u_values.opaque_wall_only)
            g_window = estimate_conductance(room.window_area_m2, u_values.window_only)
            total_ext = g_wall + g_window
            conductances[(room.room_id, "exterior")] = total_ext

        for neighbor_id in room.neighbor_ids:
            if neighbor_id == "exterior" or neighbor_id not in room_by_id:
                continue

            neighbor = room_by_id[neighbor_id]
            avg_area = (room.wall_area_m2 + neighbor.wall_area_m2) / 2
            g_internal = estimate_conductance(avg_area / 2, u_values.internal_wall)

            pair = sorted([room.room_id, neighbor_id])
            key: tuple[str, str] = (pair[0], pair[1])
            if key in conductances:
                conductances[key] = (conductances[key] + g_internal) / 2
            else:
                conductances[key] = g_internal

    return room_masses, conductances


class FirstPrinciplesParameters(TypedDict):
    """Parameters computed from first principles."""

    thermal_mass: dict[str, float]
    conductances: dict[tuple[str, str], float]


def create_parameters_from_geometry(
    rooms: list[RoomGeometry],
    u_values: UValuesNorway | None = None,
) -> ThermalParameters:
    """Create ThermalParameters from room geometry.

    This provides sensible defaults without requiring any data collection
    or parameter estimation. Useful for:
    - Initial baseline predictions
    - Quick what-if analysis
    - Validation against estimated parameters

    Args:
        rooms: List of room geometries
        u_values: U-values to use (defaults to Norwegian standards)

    Returns:
        ThermalParameters ready for prediction
    """
    if u_values is None:
        u_values = UValuesNorway()

    room_masses, conductance_dict = compute_conductance_from_rooms(rooms, u_values)

    rooms_params: dict[str, RoomParameters] = {}
    for room in rooms:
        ext_g = conductance_dict.get((room.room_id, "exterior"))
        rooms_params[room.room_id] = RoomParameters(
            thermal_mass=room_masses[room.room_id],
            exterior_conductance=ext_g,
        )

    return ThermalParameters(rooms=rooms_params, conductances=conductance_dict)
