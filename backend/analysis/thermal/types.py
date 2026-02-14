"""Core data types for thermal system identification.

This module defines all dataclasses used across the thermal analysis system.
The building is modeled as a lumped RC thermal network where:
- Nodes represent rooms (thermal capacitance C)
- Edges represent walls (thermal conductance G = 1/R)

Governing equation for node i:
    C_i * dT_i/dt = P_i + Σ_j G_ij * (T_j - T_i) + G_ext * (T_ext - T_i)
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

# -----------------------------------------------------------------------------
# Time Series Data
# -----------------------------------------------------------------------------


@dataclass
class RoomTimeSeries:
    """Historical time series data for a single room.

    All lists must have the same length. Data is assumed to be uniformly sampled.
    """

    room_id: str
    timestamps: list[datetime]
    temperature: list[float]  # °C
    heating_power: list[float]  # W

    def __len__(self) -> int:
        return len(self.timestamps)

    def to_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert to numpy arrays (temperature, heating_power)."""
        return (
            np.asarray(self.temperature, dtype=np.float64),
            np.asarray(self.heating_power, dtype=np.float64),
        )


@dataclass
class ExternalConditions:
    """External temperature conditions (constant or time-varying)."""

    temperature: float | list[float]  # °C or time series

    def at_timestep(self, t: int) -> float:
        """Get external temperature at timestep t."""
        if isinstance(self.temperature, (int, float)):
            return float(self.temperature)
        return self.temperature[t]

    def to_array(self, n_timesteps: int) -> NDArray[np.float64]:
        """Convert to numpy array of length n_timesteps."""
        if isinstance(self.temperature, (int, float)):
            return np.full(n_timesteps, float(self.temperature), dtype=np.float64)
        return np.asarray(self.temperature[:n_timesteps], dtype=np.float64)


# -----------------------------------------------------------------------------
# Thermal Network Graph
# -----------------------------------------------------------------------------


@dataclass
class EdgeSpec:
    """Specification of a thermal connection between two nodes.

    Attributes:
        node_a: First node ID
        node_b: Second node ID
        wall_area_m2: Wall area in m² (used for initial guess scaling)
        is_exterior: True if this edge connects to the exterior
    """

    node_a: str
    node_b: str
    wall_area_m2: float
    is_exterior: bool = False


@dataclass
class ThermalGraph:
    """Graph representation of a thermal network.

    Nodes are rooms, edges are thermal connections (walls).
    The exterior is a special implicit node.
    """

    node_ids: list[str]
    edges: list[EdgeSpec]
    external_node_id: str = "exterior"

    _neighbor_cache: dict[str, list[tuple[str, int]]] = field(default_factory=dict, repr=False)

    def node_index(self, node_id: str) -> int:
        """Get the index of a node in node_ids."""
        return self.node_ids.index(node_id)

    def get_neighbors(self, node_id: str) -> list[tuple[str, int]]:
        """Get (neighbor_id, edge_index) pairs for a node.

        Results are cached for performance.
        """
        if node_id in self._neighbor_cache:
            return self._neighbor_cache[node_id]

        neighbors: list[tuple[str, int]] = []
        for i, edge in enumerate(self.edges):
            if edge.node_a == node_id:
                neighbors.append((edge.node_b, i))
            elif edge.node_b == node_id:
                neighbors.append((edge.node_a, i))

        self._neighbor_cache[node_id] = neighbors
        return neighbors

    def interior_edges(self) -> list[EdgeSpec]:
        """Get all interior (non-exterior) edges."""
        return [e for e in self.edges if not e.is_exterior]

    def exterior_edges(self) -> list[EdgeSpec]:
        """Get all exterior edges."""
        return [e for e in self.edges if e.is_exterior]


# -----------------------------------------------------------------------------
# Thermal Parameters
# -----------------------------------------------------------------------------


@dataclass
class RoomParameters:
    """Estimated thermal parameters for a single room.

    Attributes:
        thermal_mass: Effective thermal capacitance (J/K)
        exterior_conductance: Conductance to outside (W/K), None if interior room
    """

    thermal_mass: float  # J/K
    exterior_conductance: float | None = None  # W/K


@dataclass
class ThermalParameters:
    """Complete set of estimated thermal parameters for a building.

    This is the output of the system identification process.
    """

    # Per-room parameters: room_id -> RoomParameters
    rooms: dict[str, RoomParameters]

    # Inter-room conductances: (room_a, room_b) -> conductance W/K
    # Keys are sorted alphabetically for consistency
    conductances: dict[tuple[str, str], float]

    def get_conductance(self, room_a: str, room_b: str) -> float:
        """Get conductance between two rooms (order-independent)."""
        key = tuple(sorted([room_a, room_b]))
        return self.conductances.get((key[0], key[1]), 0.0)


# -----------------------------------------------------------------------------
# Estimation Configuration
# -----------------------------------------------------------------------------


@dataclass
class EstimationConfig:
    """Configuration for thermal parameter estimation.

    Default values are reasonable for typical office/residential buildings.
    """

    # Time step between observations (seconds)
    dt_seconds: float = 180.0

    # Bounds for thermal mass (J/K)
    thermal_mass_min: float = 1e4  # 10 kJ/K (very small room)
    thermal_mass_max: float = 1e7  # 10 MJ/K (large room with heavy mass)

    # Bounds for conductance (W/K)
    conductance_min: float = 0.1  # Very well insulated
    conductance_max: float = 100.0  # Very conductive (or open window)

    # Initial guess scaling
    initial_thermal_mass: float = 5e5  # 500 kJ/K (typical small room)
    initial_conductance_per_m2: float = 1.5  # W/(m²·K) typical interior wall

    # Regularization weight (penalizes deviation from initial guess)
    # Set to 0 to disable, ~0.01 for light regularization
    regularization: float = 0.01

    # Loss function for outlier robustness
    # Options: 'linear' (L2), 'soft_l1', 'huber', 'cauchy', 'arctan'
    # 'huber' recommended for data with occasional door/window events
    loss: str = "huber"


# -----------------------------------------------------------------------------
# Estimation Results
# -----------------------------------------------------------------------------


@dataclass
class DataQualityWarning:
    """Warning about insufficient data variation for reliable estimation."""

    room_id: str
    issue: str  # 'low_temp_range', 'low_power_range', 'low_temp_change_variation'
    observed_value: float
    threshold: float


@dataclass
class FitMetrics:
    """Quality metrics for the estimation fit."""

    rmse: float  # Root mean square error of temperature prediction (K)
    r_squared: float  # Coefficient of determination
    n_observations: int  # Total number of observations used


@dataclass
class TimestepResidual:
    """Residual for a single timestep and room.

    Positive residual = observed temperature rose less than predicted
    (more heat loss than model predicts - possible open window/door).
    """

    timestep: int
    room_id: str
    residual: float  # dT_observed - dT_predicted
    temperature: float  # Temperature at this timestep
    heating_power: float  # Heating power at this timestep


@dataclass
class EstimationResult:
    """Complete result of thermal parameter estimation."""

    # Estimated parameters
    parameters: ThermalParameters

    # Fit quality
    fit_metrics: FitMetrics

    # Optimization outcome
    success: bool
    message: str

    # Data quality warnings
    warnings: list[DataQualityWarning]
    unreliable_rooms: set[str]

    # Per-timestep residuals (for anomaly detection)
    residuals: list[TimestepResidual]


# -----------------------------------------------------------------------------
# Anomaly Detection
# -----------------------------------------------------------------------------


@dataclass
class AnomalyDetectionConfig:
    """Configuration for thermal anomaly detection."""

    # Threshold in standard deviations for flagging anomalies
    threshold_std: float = 3.0

    # Minimum consecutive timesteps to confirm sustained anomaly
    min_consecutive: int = 2

    # Window size for computing rolling statistics
    rolling_window: int = 20


@dataclass
class ThermalAnomaly:
    """A detected thermal anomaly (e.g., window/door left open)."""

    room_id: str
    start_timestep: int
    end_timestep: int
    anomaly_type: str  # 'heat_loss' or 'heat_gain'
    magnitude: float  # Average residual magnitude during anomaly
    estimated_extra_conductance: float  # Estimated W/K of extra heat loss


@dataclass
class AnomalyReport:
    """Report of all detected anomalies."""

    anomalies: list[ThermalAnomaly]
    anomalous_timesteps_by_room: dict[str, list[int]]


# -----------------------------------------------------------------------------
# Cooling Curve Analysis
# -----------------------------------------------------------------------------


@dataclass
class CoolingCurveSegment:
    """A segment of cooling data (e.g., nighttime with heating off)."""

    room_id: str
    start_timestep: int
    end_timestep: int
    start_temp: float  # °C
    end_temp: float  # °C
    avg_external_temp: float  # °C during segment


@dataclass
class CoolingCurveResult:
    """Result of fitting an exponential cooling curve.

    T(t) = T_ext + (T_0 - T_ext) * exp(-t/τ)
    where τ = R*C is the time constant.
    """

    room_id: str
    time_constant_seconds: float  # τ = R*C
    estimated_resistance: float | None  # R in K/W (if steady state known)
    estimated_capacitance: float | None  # C in J/K (if R known)
    fit_r_squared: float


# -----------------------------------------------------------------------------
# Heat Flow Mapping
# -----------------------------------------------------------------------------


@dataclass
class HeatFlowEdge:
    """Heat flow along a single edge at a specific time."""

    from_room: str
    to_room: str
    flow_watts: float  # Positive = heat flowing from -> to
    conductance: float  # W/K


@dataclass
class HeatFlowSnapshot:
    """Heat flow state at a single timestep."""

    timestep: int
    flows: list[HeatFlowEdge]
    net_by_room: dict[str, float]  # Positive = room gaining heat


@dataclass
class ThermalBridge:
    """A thermal bridge (weak point in insulation)."""

    room_ids: tuple[str, str]  # Rooms connected by the bridge
    is_exterior: bool
    conductance: float  # W/K
    severity: str  # 'low', 'medium', 'high'
