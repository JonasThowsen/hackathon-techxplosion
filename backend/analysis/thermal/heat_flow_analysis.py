"""Heat flow analysis - estimate thermal properties from historical data.

Room -> Node
- Thermal Mass (C): J/K

Wall -> Edge
- Thermal conductivity (G): W/K (= U * A)

Goal: Estimate values based on historical:
- heating_power (W)
- temperature (°C)
- neighbor temperatures
- external temperature

Method: Nonlinear least squares optimization
Governing equation: C_i * dT_i/dt = P_i + Σ G_ij(T_j - T_i)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # pyright: ignore[reportUnknownVariableType]

from analysis.history import RoomHistory


@dataclass
class EdgeSpec:
    """Specification of an edge (wall) between nodes."""

    node_a: str
    node_b: str
    wall_area_m2: float  # Used for initial guess scaling


@dataclass
class ThermalGraph:
    """Graph structure for thermal network."""

    node_ids: list[str]
    edges: list[EdgeSpec]
    external_node_id: str = "exterior"  # Special node for outside temperature

    def node_index(self, node_id: str) -> int:
        return self.node_ids.index(node_id)

    def get_neighbors(self, node_id: str) -> list[tuple[str, int]]:
        """Get (neighbor_id, edge_index) pairs for a node."""
        neighbors: list[tuple[str, int]] = []
        for i, edge in enumerate(self.edges):
            if edge.node_a == node_id:
                neighbors.append((edge.node_b, i))
            elif edge.node_b == node_id:
                neighbors.append((edge.node_a, i))
        return neighbors


@dataclass
class EstimatedThermalProperties:
    """Result of thermal property estimation."""

    # Per-node thermal mass (J/K)
    thermal_mass: dict[str, float]

    # Per-edge conductance (W/K)
    conductance: dict[tuple[str, str], float]

    # Fit quality metrics
    rmse: float  # Root mean square error of temperature prediction (K)
    r_squared: float  # Coefficient of determination

    # Optimization result details
    success: bool
    message: str
    n_observations: int


@dataclass
class EstimationConfig:
    """Configuration for thermal property estimation."""

    # Time step between observations (seconds)
    dt_seconds: float = 180.0

    # Bounds for thermal mass (J/K) - typical room range
    thermal_mass_min: float = 1e4  # 10 kJ/K
    thermal_mass_max: float = 1e7  # 10 MJ/K

    # Bounds for conductance (W/K)
    conductance_min: float = 0.1  # Very insulated
    conductance_max: float = 100.0  # Very conductive

    # Initial guess scaling
    initial_thermal_mass: float = 5e5  # 500 kJ/K (typical small room)
    initial_conductance_per_m2: float = 1.5  # W/(m²·K) * m² = W/K

    # Regularization weight (penalizes deviation from initial guess)
    regularization: float = 0.01


@dataclass
class _OptimizeResult:
    """Typed wrapper for scipy optimization result."""

    x: NDArray[np.float64]
    success: bool
    message: str


class _ScipyOptimizeResult:
    """Type stub for scipy.optimize.OptimizeResult."""

    x: NDArray[np.float64]
    success: bool
    message: str


def _run_least_squares(
    residuals_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x0: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> _OptimizeResult:
    """Run scipy least_squares with proper typing."""

    # Call with full kwargs, cast result
    result = cast(
        _ScipyOptimizeResult,
        least_squares(
            residuals_fn,
            x0,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=10000,
        ),
    )

    return _OptimizeResult(
        x=np.asarray(result.x, dtype=np.float64),
        success=result.success,
        message=result.message,
    )


def estimate_thermal_properties(
    histories: dict[str, RoomHistory],
    graph: ThermalGraph,
    external_temp: float | list[float],
    config: EstimationConfig | None = None,
) -> EstimatedThermalProperties:
    """Estimate thermal conductivity and mass from historical data.

    Uses nonlinear least squares to find parameters that best explain
    the observed temperature dynamics.

    Args:
        histories: Historical data per node (room)
        graph: Thermal network topology
        external_temp: Outside temperature (constant or time series)
        config: Estimation configuration

    Returns:
        Estimated thermal properties with fit quality metrics
    """
    if config is None:
        config = EstimationConfig()

    n_nodes = len(graph.node_ids)
    n_edges = len(graph.edges)
    n_params = n_nodes + n_edges  # C_i for each node, G_ij for each edge

    # Validate inputs
    if not histories:
        raise ValueError("No historical data provided")

    # Get consistent time series length
    n_timesteps = min(len(h.temperature) for h in histories.values())
    if n_timesteps < 2:
        raise ValueError("Need at least 2 timesteps for estimation")

    # Build observation matrices
    # For each timestep t, we have the equation:
    # C_i * (T_i(t+1) - T_i(t))/dt = P_i(t) + Σ G_ij(T_j(t) - T_i(t))

    # Convert to numpy arrays
    temp_matrix = np.zeros((n_nodes, n_timesteps))
    power_matrix = np.zeros((n_nodes, n_timesteps))

    for i, node_id in enumerate(graph.node_ids):
        if node_id not in histories:
            raise ValueError(f"No history for node {node_id}")
        h = histories[node_id]
        temp_matrix[i, :] = h.temperature[:n_timesteps]
        power_matrix[i, :] = h.heating_power[:n_timesteps]

    # External temperature (broadcast to time series if constant)
    if isinstance(external_temp, (int, float)):
        ext_temp_array = np.full(n_timesteps, external_temp)
    else:
        ext_temp_array = np.array(external_temp[:n_timesteps])

    # Build initial guess
    x0 = np.zeros(n_params)
    # Thermal masses
    x0[:n_nodes] = config.initial_thermal_mass
    # Conductances (scaled by wall area)
    for j, edge in enumerate(graph.edges):
        x0[n_nodes + j] = config.initial_conductance_per_m2 * edge.wall_area_m2

    # Build bounds
    lower_bounds = np.zeros(n_params)
    upper_bounds = np.zeros(n_params)
    lower_bounds[:n_nodes] = config.thermal_mass_min
    upper_bounds[:n_nodes] = config.thermal_mass_max
    lower_bounds[n_nodes:] = config.conductance_min
    upper_bounds[n_nodes:] = config.conductance_max

    # Capture config for closure
    dt_seconds = config.dt_seconds
    regularization = config.regularization
    initial_thermal_mass = config.initial_thermal_mass
    initial_conductance_per_m2 = config.initial_conductance_per_m2

    def residuals(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute residuals for all observations."""
        thermal_masses = x[:n_nodes]
        conductances = x[n_nodes:]

        resid_list: list[float] = []

        # For each timestep (except last)
        for t in range(n_timesteps - 1):
            # For each node
            for i, node_id in enumerate(graph.node_ids):
                # Observed temperature change
                dT_observed = float(temp_matrix[i, t + 1] - temp_matrix[i, t])

                # Heat input from HVAC
                P_i = float(power_matrix[i, t])

                # Heat flow from neighbors
                Q_neighbors = 0.0
                for neighbor_id, edge_idx in graph.get_neighbors(node_id):
                    G_ij = float(conductances[edge_idx])
                    if neighbor_id == graph.external_node_id:
                        T_neighbor = float(ext_temp_array[t])
                    else:
                        neighbor_idx = graph.node_index(neighbor_id)
                        T_neighbor = float(temp_matrix[neighbor_idx, t])
                    Q_neighbors += G_ij * (T_neighbor - float(temp_matrix[i, t]))

                # Predicted temperature change
                C_i = float(thermal_masses[i])
                dT_predicted = (P_i + Q_neighbors) * dt_seconds / C_i

                # Residual
                resid_list.append(dT_observed - dT_predicted)

        # Add regularization (soft penalty for deviating from initial guess)
        if regularization > 0:
            reg_weight = regularization
            for i in range(n_nodes):
                resid_list.append(reg_weight * (float(x[i]) - float(x0[i])) / initial_thermal_mass)
            for j in range(n_edges):
                resid_list.append(
                    reg_weight
                    * (float(x[n_nodes + j]) - float(x0[n_nodes + j]))
                    / (initial_conductance_per_m2 * graph.edges[j].wall_area_m2)
                )

        return np.array(resid_list)

    # Run optimization
    result = _run_least_squares(residuals, x0, lower_bounds, upper_bounds)

    # Extract results
    result_x = result.x
    thermal_mass_dict: dict[str, float] = {}
    for i, node_id in enumerate(graph.node_ids):
        thermal_mass_dict[node_id] = float(result_x[i])

    conductance_dict: dict[tuple[str, str], float] = {}
    for j, edge in enumerate(graph.edges):
        key = (edge.node_a, edge.node_b)
        conductance_dict[key] = float(result_x[n_nodes + j])

    # Compute fit quality
    final_residuals = residuals(result_x)
    # Remove regularization residuals for RMSE calculation
    n_obs = (n_timesteps - 1) * n_nodes
    obs_residuals = final_residuals[:n_obs]
    rmse = float(np.sqrt(np.mean(obs_residuals**2)))

    # R-squared
    dT_observed_all: list[float] = []
    for t in range(n_timesteps - 1):
        for i in range(n_nodes):
            dT_observed_all.append(float(temp_matrix[i, t + 1] - temp_matrix[i, t]))
    dT_observed_arr = np.array(dT_observed_all)
    ss_res = float(np.sum(obs_residuals**2))
    ss_tot = float(np.sum((dT_observed_arr - np.mean(dT_observed_arr)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return EstimatedThermalProperties(
        thermal_mass=thermal_mass_dict,
        conductance=conductance_dict,
        rmse=rmse,
        r_squared=r_squared,
        success=result.success,
        message=result.message,
        n_observations=n_obs,
    )


def build_graph_from_adjacency(
    room_ids: list[str],
    adjacency: dict[str, list[str]],
    wall_areas: dict[tuple[str, str], float] | None = None,
    exterior_rooms: set[str] | None = None,
    default_wall_area: float = 9.0,  # 3m x 3m default
) -> ThermalGraph:
    """Build ThermalGraph from adjacency information.

    Args:
        room_ids: List of room IDs
        adjacency: Adjacency dict {room_id: [neighbor_ids]}
        wall_areas: Optional wall areas {(room_a, room_b): area_m2}
        exterior_rooms: Rooms with exterior walls
        default_wall_area: Default wall area if not specified

    Returns:
        ThermalGraph for estimation
    """
    edges: list[EdgeSpec] = []
    seen: set[tuple[str, str]] = set()

    # Interior edges
    for room_id, neighbors in adjacency.items():
        for neighbor_id in neighbors:
            if neighbor_id not in room_ids:
                continue
            # Create sorted pair with explicit tuple type
            sorted_ids = sorted([room_id, neighbor_id])
            pair: tuple[str, str] = (sorted_ids[0], sorted_ids[1])
            if pair in seen:
                continue
            seen.add(pair)

            area = default_wall_area
            if wall_areas:
                reverse_pair: tuple[str, str] = (pair[1], pair[0])
                area = wall_areas.get(pair, wall_areas.get(reverse_pair, default_wall_area))

            edges.append(EdgeSpec(node_a=pair[0], node_b=pair[1], wall_area_m2=area))

    # Exterior edges
    if exterior_rooms:
        for room_id in exterior_rooms:
            if room_id in room_ids:
                edges.append(EdgeSpec(node_a=room_id, node_b="exterior", wall_area_m2=default_wall_area))

    return ThermalGraph(node_ids=room_ids, edges=edges)
