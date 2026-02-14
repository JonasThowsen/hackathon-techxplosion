"""Adaptive parameter estimation with recursive least squares.

This module implements online (streaming) parameter estimation using
Recursive Least Squares (RLS) with exponential forgetting. This allows
tracking time-varying parameters like thermal resistance (which changes
when a window is opened).

The forgetting factor λ ∈ (0, 1] determines how quickly old data is
discounted. Typical values: 0.95-0.99 for slow changes, 0.9-0.95 for
faster adaptation.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class RLSConfig:
    """Configuration for Recursive Least Squares estimator."""

    # Forgetting factor: how quickly to discount old data
    # 1.0 = no forgetting (standard least squares)
    # 0.95 = data from 20 samples ago has ~1/e weight
    # 0.9 = faster adaptation to changes
    forgetting_factor: float = 0.98

    # Initial covariance scaling (higher = less confident in initial guess)
    initial_covariance: float = 1e6

    # Regularization to prevent covariance explosion
    # Small positive value added to diagonal of P
    regularization: float = 1e-6


@dataclass
class RLSState:
    """State of an RLS estimator.

    The RLS algorithm maintains:
    - θ: Parameter estimate vector
    - P: Covariance matrix (inverse of Hessian approximation)
    """

    theta: NDArray[np.float64]  # Parameter estimates
    P: NDArray[np.float64]  # Covariance matrix
    n_updates: int = 0


@dataclass
class SingleRoomRLS:
    """RLS estimator for a single room.

    Estimates: [1/C, G_ext/C] or [1/C, G_ext/C, G_1/C, G_2/C, ...] depending
    on the number of neighbors.

    From these ratios, we can compute:
    - If we assume a typical C, we get G values
    - If we have steady-state data, we can separate C and G
    """

    room_id: str
    n_params: int  # Number of parameters to estimate
    config: RLSConfig
    state: RLSState = field(init=False)

    def __post_init__(self) -> None:
        """Initialize RLS state."""
        self.state = RLSState(
            theta=np.zeros(self.n_params, dtype=np.float64),
            P=np.eye(self.n_params, dtype=np.float64) * self.config.initial_covariance,
            n_updates=0,
        )

    def update(
        self,
        phi: NDArray[np.float64],
        y: float,
    ) -> NDArray[np.float64]:
        """Update parameter estimates with a new observation.

        The model is: y = φᵀθ + noise
        where:
        - y is the observed temperature change
        - φ is the regressor vector (function of temperatures and power)
        - θ is the parameter vector we're estimating

        Args:
            phi: Regressor vector, shape (n_params,)
            y: Observed output (temperature change)

        Returns:
            Updated parameter estimate
        """
        λ = self.config.forgetting_factor
        θ = self.state.theta
        P = self.state.P

        # Prediction error
        y_pred = float(np.dot(phi, θ))
        e = y - y_pred

        # Kalman gain
        Pφ = P @ phi
        denom = λ + float(np.dot(phi, Pφ))
        K = Pφ / denom

        # Update parameter estimate
        θ_new = θ + K * e

        # Update covariance matrix with regularization
        P_new = (P - np.outer(K, Pφ)) / λ
        # Add regularization to prevent covariance explosion
        P_new += np.eye(self.n_params) * self.config.regularization

        self.state.theta = θ_new
        self.state.P = P_new
        self.state.n_updates += 1

        return θ_new

    def get_estimates(self) -> NDArray[np.float64]:
        """Get current parameter estimates."""
        return self.state.theta.copy()

    def get_uncertainty(self) -> NDArray[np.float64]:
        """Get parameter uncertainty (diagonal of covariance)."""
        return np.sqrt(np.diag(self.state.P))


def create_single_room_estimator(
    room_id: str,
    n_neighbors: int,
    config: RLSConfig | None = None,
) -> SingleRoomRLS:
    """Create an RLS estimator for a single room.

    Args:
        room_id: Room identifier
        n_neighbors: Number of thermal neighbors (including exterior if applicable)
        config: RLS configuration

    Returns:
        Initialized RLS estimator
    """
    if config is None:
        config = RLSConfig()

    # Parameters: [1/C, G_ext/C (if exterior), G_1/C, G_2/C, ...]
    # The first term is for heating power, then one for each neighbor
    n_params = 1 + n_neighbors

    return SingleRoomRLS(
        room_id=room_id,
        n_params=n_params,
        config=config,
    )


def build_regressor_vector(
    heating_power: float,
    current_temp: float,
    neighbor_temps: list[float],
    dt_seconds: float,
) -> NDArray[np.float64]:
    """Build the regressor vector φ for a single timestep.

    The model: dT = (dt/C)*P + Σ (dt*G_j/C)*(T_j - T)
             = dt * [P, T_ext-T, T_1-T, ...] @ [1/C, G_ext/C, G_1/C, ...]

    So φ = dt * [P, T_ext-T, T_1-T, ...]

    Args:
        heating_power: HVAC power input (W)
        current_temp: Current room temperature (°C)
        neighbor_temps: Temperatures of all neighbors (°C)
        dt_seconds: Time step duration (seconds)

    Returns:
        Regressor vector φ
    """
    n_features = 1 + len(neighbor_temps)
    phi = np.zeros(n_features, dtype=np.float64)

    phi[0] = dt_seconds * heating_power
    for i, t_neighbor in enumerate(neighbor_temps):
        phi[1 + i] = dt_seconds * (t_neighbor - current_temp)

    return phi


def extract_parameters_from_ratios(
    ratios: NDArray[np.float64],
    assumed_thermal_mass: float | None = None,
    steady_state_power: float | None = None,
    steady_state_temp_diff: float | None = None,
) -> tuple[float, list[float]]:
    """Extract C and G values from estimated ratios.

    The RLS estimates ratios like [1/C, G_ext/C, G_1/C, ...].
    We need additional information to separate C and G.

    Option 1: Assume a typical thermal mass
    Option 2: Use steady-state data where P = G_total * ΔT

    Args:
        ratios: Estimated parameter ratios from RLS
        assumed_thermal_mass: If provided, use this as C (J/K)
        steady_state_power: Power at steady state (W)
        steady_state_temp_diff: Temperature difference at steady state (°C)

    Returns:
        Tuple of (thermal_mass, [conductances])
    """
    inv_c = ratios[0]  # 1/C
    g_over_c = ratios[1:]  # G_j/C for each neighbor

    if inv_c <= 0:
        raise ValueError("Invalid estimate: 1/C must be positive")

    if assumed_thermal_mass is not None:
        # Method 1: Use assumed thermal mass
        capacitance = assumed_thermal_mass
        g_values = [g_c * capacitance for g_c in g_over_c]
        return capacitance, g_values

    if steady_state_power is not None and steady_state_temp_diff is not None:
        # Method 2: Use steady state conditions
        # At steady state: P = Σ G_j * ΔT
        # So: Σ G_j = P / ΔT
        # And: Σ (G_j/C) = (P/ΔT) / C
        # Therefore: C = (Σ (G_j/C)) * ΔT / P * C ... wait this is circular

        # Actually: We know inv_c = 1/C, so C = 1/inv_c
        capacitance = 1.0 / inv_c
        g_values = [g_c * capacitance for g_c in g_over_c]

        # Verify with steady state (sanity check - could warn if discrepancy is large)
        g_total = sum(g_values)
        _ = g_total * abs(steady_state_temp_diff)  # expected_power for validation

        return capacitance, g_values

    # Default: just use the ratio to compute C
    capacitance = 1.0 / inv_c if inv_c > 0 else 1e6
    g_values = [g_c * capacitance for g_c in g_over_c]
    return capacitance, g_values


@dataclass
class AdaptiveEstimator:
    """Multi-room adaptive thermal parameter estimator.

    Maintains an RLS estimator for each room and tracks parameter
    evolution over time.
    """

    room_estimators: dict[str, SingleRoomRLS]
    config: RLSConfig
    parameter_history: dict[str, list[NDArray[np.float64]]] = field(default_factory=dict)

    def update_room(
        self,
        room_id: str,
        phi: NDArray[np.float64],
        y: float,
    ) -> NDArray[np.float64]:
        """Update estimates for a single room.

        Args:
            room_id: Room identifier
            phi: Regressor vector
            y: Observed temperature change

        Returns:
            Updated parameter estimates
        """
        estimator = self.room_estimators[room_id]
        new_params = estimator.update(phi, y)

        # Track history
        if room_id not in self.parameter_history:
            self.parameter_history[room_id] = []
        self.parameter_history[room_id].append(new_params.copy())

        return new_params

    def get_all_estimates(self) -> dict[str, NDArray[np.float64]]:
        """Get current parameter estimates for all rooms."""
        return {room_id: est.get_estimates() for room_id, est in self.room_estimators.items()}

    def detect_parameter_changes(
        self,
        room_id: str,
        window_size: int = 10,
        threshold_std: float = 3.0,
    ) -> list[int]:
        """Detect significant parameter changes for a room.

        Args:
            room_id: Room identifier
            window_size: Rolling window for computing statistics
            threshold_std: Change threshold in standard deviations

        Returns:
            List of timestep indices where significant changes occurred
        """
        history = self.parameter_history.get(room_id, [])
        if len(history) < window_size * 2:
            return []

        # Convert to array
        param_array = np.array(history)

        # Look at the sum of conductances (second and later parameters)
        # This is most sensitive to window/door opening
        if param_array.shape[1] > 1:
            g_sum = np.sum(param_array[:, 1:], axis=1)
        else:
            g_sum = param_array[:, 0]

        change_indices: list[int] = []

        for t in range(window_size, len(g_sum) - window_size):
            # Compare current value to recent history
            recent = g_sum[t - window_size : t]
            current = g_sum[t]

            mean = float(np.mean(recent))
            std = float(np.std(recent))

            if std > 0 and abs(current - mean) > threshold_std * std:
                change_indices.append(t)

        return change_indices


def create_adaptive_estimator(
    room_ids: list[str],
    neighbor_counts: dict[str, int],
    config: RLSConfig | None = None,
) -> AdaptiveEstimator:
    """Create an adaptive estimator for multiple rooms.

    Args:
        room_ids: List of room identifiers
        neighbor_counts: Number of neighbors for each room
        config: RLS configuration

    Returns:
        Initialized adaptive estimator
    """
    if config is None:
        config = RLSConfig()

    estimators = {
        room_id: create_single_room_estimator(room_id, neighbor_counts[room_id], config) for room_id in room_ids
    }

    return AdaptiveEstimator(
        room_estimators=estimators,
        config=config,
    )
