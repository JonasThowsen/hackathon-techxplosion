"""Parameter estimation for thermal networks via nonlinear least squares.

This module estimates thermal mass (C) and conductance (G) parameters
from historical temperature and power data.

The estimation minimizes the error between observed and predicted
temperature changes using scipy's trust region reflective algorithm.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares  # pyright: ignore[reportUnknownVariableType]

from analysis.thermal.types import (
    DataQualityWarning,
    EstimationConfig,
    EstimationResult,
    ExternalConditions,
    FitMetrics,
    RoomParameters,
    RoomTimeSeries,
    ThermalGraph,
    ThermalParameters,
    TimestepResidual,
)

# -----------------------------------------------------------------------------
# Internal Types
# -----------------------------------------------------------------------------


class _ScipyOptimizeResult:
    """Type stub for scipy.optimize.OptimizeResult."""

    x: NDArray[np.float64]
    success: bool
    message: str


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def estimate_parameters(
    room_data: dict[str, RoomTimeSeries],
    graph: ThermalGraph,
    external: ExternalConditions,
    config: EstimationConfig | None = None,
) -> EstimationResult:
    """Estimate thermal parameters from historical data.

    This is the main entry point for thermal system identification.
    Uses nonlinear least squares to find parameters (thermal mass C
    and conductance G values) that best explain the observed
    temperature dynamics.

    Args:
        room_data: Historical time series data per room
        graph: Thermal network topology
        external: External temperature conditions
        config: Estimation configuration (uses defaults if None)

    Returns:
        EstimationResult with estimated parameters and diagnostics
    """
    if config is None:
        config = EstimationConfig()

    # Validate inputs
    _validate_inputs(room_data, graph)

    # Build data matrices
    matrices = _build_data_matrices(room_data, graph, external, config)

    # Check data quality
    warnings, unreliable = _check_data_quality(room_data, graph, config)

    # Build initial guess and bounds
    x0, lower, upper = _build_initial_guess_and_bounds(graph, config)

    # Run optimization
    result_x, success, message = _run_optimization(
        matrices=matrices,
        graph=graph,
        x0=x0,
        lower_bounds=lower,
        upper_bounds=upper,
        config=config,
    )

    # Extract parameters from solution vector
    parameters = _extract_parameters(result_x, graph)

    # Compute residuals and fit metrics
    residuals, fit_metrics = _compute_residuals_and_metrics(result_x, matrices, graph, config)

    return EstimationResult(
        parameters=parameters,
        fit_metrics=fit_metrics,
        success=success,
        message=message,
        warnings=warnings,
        unreliable_rooms=unreliable,
        residuals=residuals,
    )


# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------


class _DataMatrices:
    """Processed data matrices for optimization."""

    def __init__(
        self,
        temp_matrix: NDArray[np.float64],
        power_matrix: NDArray[np.float64],
        external_temps: NDArray[np.float64],
        node_ids: list[str],
        n_timesteps: int,
    ) -> None:
        self.temp_matrix = temp_matrix  # (n_nodes, n_timesteps)
        self.power_matrix = power_matrix  # (n_nodes, n_timesteps)
        self.external_temps = external_temps  # (n_timesteps,)
        self.node_ids = node_ids
        self.n_timesteps = n_timesteps


def _validate_inputs(
    room_data: dict[str, RoomTimeSeries],
    graph: ThermalGraph,
) -> None:
    """Validate input data."""
    if not room_data:
        raise ValueError("No room data provided")

    for node_id in graph.node_ids:
        if node_id not in room_data:
            raise ValueError(f"No data for node {node_id}")

    n_timesteps = min(len(data) for data in room_data.values())
    if n_timesteps < 2:
        raise ValueError("Need at least 2 timesteps for estimation")


def _build_data_matrices(
    room_data: dict[str, RoomTimeSeries],
    graph: ThermalGraph,
    external: ExternalConditions,
    config: EstimationConfig,
) -> _DataMatrices:
    """Build numpy matrices from room data."""
    _ = config  # Reserved for filtering/preprocessing options
    n_nodes = len(graph.node_ids)
    n_timesteps = min(len(data) for data in room_data.values())

    temp_matrix = np.zeros((n_nodes, n_timesteps), dtype=np.float64)
    power_matrix = np.zeros((n_nodes, n_timesteps), dtype=np.float64)

    for i, node_id in enumerate(graph.node_ids):
        data = room_data[node_id]
        temp_matrix[i, :] = data.temperature[:n_timesteps]
        power_matrix[i, :] = data.heating_power[:n_timesteps]

    external_temps = external.to_array(n_timesteps)

    return _DataMatrices(
        temp_matrix=temp_matrix,
        power_matrix=power_matrix,
        external_temps=external_temps,
        node_ids=graph.node_ids,
        n_timesteps=n_timesteps,
    )


# -----------------------------------------------------------------------------
# Data Quality
# -----------------------------------------------------------------------------


def _check_data_quality(
    room_data: dict[str, RoomTimeSeries],
    graph: ThermalGraph,
    config: EstimationConfig,
) -> tuple[list[DataQualityWarning], set[str]]:
    """Check data quality and return warnings."""
    _ = config  # Reserved for configurable thresholds
    warnings: list[DataQualityWarning] = []
    unreliable: set[str] = set()

    # Thresholds
    min_temp_range = 1.0  # °C
    min_power_range = 200.0  # W
    min_temp_change_std = 0.05  # °C

    for node_id in graph.node_ids:
        data = room_data[node_id]
        temps = np.array(data.temperature)
        powers = np.array(data.heating_power)

        temp_range = float(np.max(temps) - np.min(temps))
        power_range = float(np.max(powers) - np.min(powers))
        temp_changes = np.diff(temps)
        temp_change_std = float(np.std(temp_changes))

        if temp_range < min_temp_range:
            warnings.append(
                DataQualityWarning(
                    room_id=node_id,
                    issue="low_temp_range",
                    observed_value=temp_range,
                    threshold=min_temp_range,
                )
            )
            unreliable.add(node_id)

        if power_range < min_power_range:
            warnings.append(
                DataQualityWarning(
                    room_id=node_id,
                    issue="low_power_range",
                    observed_value=power_range,
                    threshold=min_power_range,
                )
            )
            unreliable.add(node_id)

        if temp_change_std < min_temp_change_std:
            warnings.append(
                DataQualityWarning(
                    room_id=node_id,
                    issue="low_temp_change_variation",
                    observed_value=temp_change_std,
                    threshold=min_temp_change_std,
                )
            )
            unreliable.add(node_id)

    return warnings, unreliable


# -----------------------------------------------------------------------------
# Optimization Setup
# -----------------------------------------------------------------------------


def _build_initial_guess_and_bounds(
    graph: ThermalGraph,
    config: EstimationConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Build initial guess and parameter bounds.

    Parameter vector layout:
        [C_0, C_1, ..., C_n, G_0, G_1, ..., G_m]
    where n = number of nodes, m = number of edges
    """
    n_nodes = len(graph.node_ids)
    n_edges = len(graph.edges)
    n_params = n_nodes + n_edges

    x0 = np.zeros(n_params, dtype=np.float64)
    lower = np.zeros(n_params, dtype=np.float64)
    upper = np.zeros(n_params, dtype=np.float64)

    # Thermal masses
    x0[:n_nodes] = config.initial_thermal_mass
    lower[:n_nodes] = config.thermal_mass_min
    upper[:n_nodes] = config.thermal_mass_max

    # Conductances (scaled by wall area)
    for j, edge in enumerate(graph.edges):
        x0[n_nodes + j] = config.initial_conductance_per_m2 * edge.wall_area_m2
        lower[n_nodes + j] = config.conductance_min
        upper[n_nodes + j] = config.conductance_max

    return x0, lower, upper


def _run_optimization(
    matrices: _DataMatrices,
    graph: ThermalGraph,
    x0: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    config: EstimationConfig,
) -> tuple[NDArray[np.float64], bool, str]:
    """Run the least squares optimization."""
    # Build residual function
    residual_fn = _build_residual_function(matrices, graph, x0, config)

    # Run optimization
    result = cast(
        _ScipyOptimizeResult,
        least_squares(
            residual_fn,
            x0,
            bounds=(lower_bounds, upper_bounds),
            method="trf",
            loss=config.loss,
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=10000,
        ),
    )

    return (
        np.asarray(result.x, dtype=np.float64),
        result.success,
        result.message,
    )


def _build_residual_function(
    matrices: _DataMatrices,
    graph: ThermalGraph,
    x0: NDArray[np.float64],
    config: EstimationConfig,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Build the residual function for optimization.

    The residual function computes the difference between observed
    and predicted temperature changes for all timesteps and rooms.
    """
    n_nodes = len(graph.node_ids)
    n_edges = len(graph.edges)
    n_timesteps = matrices.n_timesteps
    dt_seconds = config.dt_seconds
    regularization = config.regularization

    # Pre-compute neighbor lookup
    neighbor_info: list[list[tuple[str, int]]] = [graph.get_neighbors(node_id) for node_id in graph.node_ids]

    def residuals(x: NDArray[np.float64]) -> NDArray[np.float64]:
        thermal_masses = x[:n_nodes]
        conductances = x[n_nodes:]

        resid_list: list[float] = []

        for t in range(n_timesteps - 1):
            for i, _ in enumerate(graph.node_ids):
                # Observed temperature change
                dt_observed = float(matrices.temp_matrix[i, t + 1] - matrices.temp_matrix[i, t])

                # Heat input
                p_i = float(matrices.power_matrix[i, t])

                # Heat flow from neighbors
                q_neighbors = 0.0
                for neighbor_id, edge_idx in neighbor_info[i]:
                    g_ij = float(conductances[edge_idx])

                    if neighbor_id == graph.external_node_id:
                        t_neighbor = float(matrices.external_temps[t])
                    else:
                        neighbor_idx = graph.node_index(neighbor_id)
                        t_neighbor = float(matrices.temp_matrix[neighbor_idx, t])

                    q_neighbors += g_ij * (t_neighbor - matrices.temp_matrix[i, t])

                # Predicted temperature change
                c_i = float(thermal_masses[i])
                dt_predicted = (p_i + q_neighbors) * dt_seconds / c_i

                resid_list.append(dt_observed - dt_predicted)

        # Add regularization
        if regularization > 0:
            for i in range(n_nodes):
                resid_list.append(regularization * (float(x[i]) - float(x0[i])) / config.initial_thermal_mass)
            for j in range(n_edges):
                initial_g = config.initial_conductance_per_m2 * graph.edges[j].wall_area_m2
                resid_list.append(regularization * (float(x[n_nodes + j]) - float(x0[n_nodes + j])) / initial_g)

        return np.array(resid_list)

    return residuals


# -----------------------------------------------------------------------------
# Result Extraction
# -----------------------------------------------------------------------------


def _extract_parameters(
    x: NDArray[np.float64],
    graph: ThermalGraph,
) -> ThermalParameters:
    """Extract ThermalParameters from optimization result vector."""
    n_nodes = len(graph.node_ids)

    # Build room parameters
    rooms: dict[str, RoomParameters] = {}
    for i, node_id in enumerate(graph.node_ids):
        thermal_mass = float(x[i])

        # Check for exterior connection
        ext_conductance: float | None = None
        for neighbor_id, edge_idx in graph.get_neighbors(node_id):
            if neighbor_id == graph.external_node_id:
                ext_conductance = float(x[n_nodes + edge_idx])
                break

        rooms[node_id] = RoomParameters(
            thermal_mass=thermal_mass,
            exterior_conductance=ext_conductance,
        )

    # Build conductance dict (interior edges only)
    conductances: dict[tuple[str, str], float] = {}
    for j, edge in enumerate(graph.edges):
        if edge.is_exterior:
            continue
        key = tuple(sorted([edge.node_a, edge.node_b]))
        conductances[(key[0], key[1])] = float(x[n_nodes + j])

    return ThermalParameters(rooms=rooms, conductances=conductances)


def _compute_residuals_and_metrics(
    x: NDArray[np.float64],
    matrices: _DataMatrices,
    graph: ThermalGraph,
    config: EstimationConfig,
) -> tuple[list[TimestepResidual], FitMetrics]:
    """Compute per-timestep residuals and fit metrics."""
    n_nodes = len(graph.node_ids)
    n_timesteps = matrices.n_timesteps
    dt_seconds = config.dt_seconds

    thermal_masses = x[:n_nodes]
    conductances = x[n_nodes:]

    residuals: list[TimestepResidual] = []
    all_observed: list[float] = []
    all_residuals: list[float] = []

    for t in range(n_timesteps - 1):
        for i, node_id in enumerate(graph.node_ids):
            dt_observed = float(matrices.temp_matrix[i, t + 1] - matrices.temp_matrix[i, t])
            all_observed.append(dt_observed)

            p_i = float(matrices.power_matrix[i, t])
            q_neighbors = 0.0

            for neighbor_id, edge_idx in graph.get_neighbors(node_id):
                g_ij = float(conductances[edge_idx])
                if neighbor_id == graph.external_node_id:
                    t_neighbor = float(matrices.external_temps[t])
                else:
                    neighbor_idx = graph.node_index(neighbor_id)
                    t_neighbor = float(matrices.temp_matrix[neighbor_idx, t])
                q_neighbors += g_ij * (t_neighbor - matrices.temp_matrix[i, t])

            c_i = float(thermal_masses[i])
            dt_predicted = (p_i + q_neighbors) * dt_seconds / c_i
            residual = dt_observed - dt_predicted
            all_residuals.append(residual)

            residuals.append(
                TimestepResidual(
                    timestep=t,
                    room_id=node_id,
                    residual=residual,
                    temperature=float(matrices.temp_matrix[i, t]),
                    heating_power=p_i,
                )
            )

    # Compute fit metrics
    resid_arr = np.array(all_residuals)
    obs_arr = np.array(all_observed)

    rmse = float(np.sqrt(np.mean(resid_arr**2)))

    ss_res = float(np.sum(resid_arr**2))
    ss_tot = float(np.sum((obs_arr - np.mean(obs_arr)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fit_metrics = FitMetrics(
        rmse=rmse,
        r_squared=r_squared,
        n_observations=len(all_residuals),
    )

    return residuals, fit_metrics
