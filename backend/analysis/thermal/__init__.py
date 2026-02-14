"""Thermal system identification module.

This module provides tools for estimating thermal parameters (mass and
conductance) of a building from historical temperature and power data.

The building is modeled as a lumped RC thermal network where:
- Rooms are nodes with thermal capacitance (mass)
- Walls are edges with thermal conductance

Key capabilities:
- Parameter estimation via nonlinear least squares
- Adaptive online estimation with forgetting factor
- Anomaly detection (window/door openings)
- Cooling curve analysis for parameter separation
- Data quality assessment

Example usage:

    from analysis.thermal import (
        estimate_parameters,
        build_graph_from_adjacency,
        RoomTimeSeries,
        ExternalConditions,
        EstimationConfig,
    )

    # Build thermal graph
    graph = build_graph_from_adjacency(
        room_ids=["room_a", "room_b"],
        adjacency={"room_a": ["room_b"], "room_b": ["room_a"]},
        exterior_rooms={"room_a"},
    )

    # Prepare data
    room_data = {
        "room_a": RoomTimeSeries(
            room_id="room_a",
            timestamps=timestamps,
            temperature=temps_a,
            heating_power=powers_a,
        ),
        "room_b": RoomTimeSeries(...),
    }

    # Run estimation
    result = estimate_parameters(
        room_data=room_data,
        graph=graph,
        external=ExternalConditions(temperature=5.0),
    )

    print(f"R-squared: {result.fit_metrics.r_squared:.3f}")
"""

# Core types
# Adaptive estimation
from analysis.thermal.adaptive import (
    AdaptiveEstimator,
    RLSConfig,
    RLSState,
    SingleRoomRLS,
    build_regressor_vector,
    create_adaptive_estimator,
    create_single_room_estimator,
    extract_parameters_from_ratios,
)

# Anomaly detection
from analysis.thermal.anomaly import (
    RollingAnomalyDetector,
    classify_anomaly_severity,
    detect_anomalies_from_residuals,
    detect_sustained_events,
    find_recurring_patterns,
    summarize_anomalies,
)

# Cooling curve analysis
from analysis.thermal.cooling_curve import (
    IdentifiabilityReport,
    analyze_identifiability,
    estimate_from_multiple_segments,
    find_cooling_segments,
    find_nighttime_cooling,
    fit_cooling_curve,
    separate_mass_and_resistance,
)

# Data quality
from analysis.thermal.data_quality import (
    DataQualityMetrics,
    EstimationReadiness,
    QualityLevel,
    QualityThresholds,
    assess_building_data_quality,
    assess_room_data_quality,
    check_collinearity,
    check_estimation_readiness,
    compute_signal_to_noise,
)

# Parameter estimation
from analysis.thermal.estimation import (
    estimate_parameters,
)

# Graph construction
from analysis.thermal.graph import (
    build_conductance_matrix,
    build_graph_from_adjacency,
    compute_heat_flow,
    compute_heat_flow_snapshot,
    compute_heat_flow_trajectory,
    identify_thermal_bridges,
    rank_rooms_by_heat_loss,
)

# RC model
from analysis.thermal.rc_model import (
    BuildingPrediction,
    build_regression_matrices,
    compute_steady_state_temperature,
    compute_time_constant,
    estimate_resistance_from_steady_state,
    predict_building_temperature,
    predict_temperature_change,
    predict_temperature_trajectory,
)
from analysis.thermal.types import (
    AnomalyDetectionConfig,
    AnomalyReport,
    CoolingCurveResult,
    # Cooling curves
    CoolingCurveSegment,
    # Results
    DataQualityWarning,
    # Graph
    EdgeSpec,
    # Configuration
    EstimationConfig,
    EstimationResult,
    ExternalConditions,
    FitMetrics,
    # Heat flow
    HeatFlowEdge,
    HeatFlowSnapshot,
    # Parameters
    RoomParameters,
    # Time series
    RoomTimeSeries,
    # Anomalies
    ThermalAnomaly,
    ThermalBridge,
    ThermalGraph,
    ThermalParameters,
    TimestepResidual,
)

__all__ = [
    "AdaptiveEstimator",
    "AnomalyDetectionConfig",
    "AnomalyReport",
    "BuildingPrediction",
    "CoolingCurveResult",
    "CoolingCurveSegment",
    "DataQualityMetrics",
    "DataQualityWarning",
    "EdgeSpec",
    "EstimationConfig",
    "EstimationReadiness",
    "EstimationResult",
    "ExternalConditions",
    "FitMetrics",
    "HeatFlowEdge",
    "HeatFlowSnapshot",
    "IdentifiabilityReport",
    "QualityLevel",
    "QualityThresholds",
    "RLSConfig",
    "RLSState",
    "RollingAnomalyDetector",
    "RoomParameters",
    "RoomTimeSeries",
    "SingleRoomRLS",
    "ThermalAnomaly",
    "ThermalBridge",
    "ThermalGraph",
    "ThermalParameters",
    "TimestepResidual",
    "analyze_identifiability",
    "assess_building_data_quality",
    "assess_room_data_quality",
    "build_conductance_matrix",
    "build_graph_from_adjacency",
    "build_regression_matrices",
    "build_regressor_vector",
    "check_collinearity",
    "check_estimation_readiness",
    "classify_anomaly_severity",
    "compute_heat_flow",
    "compute_heat_flow_snapshot",
    "compute_heat_flow_trajectory",
    "compute_signal_to_noise",
    "compute_steady_state_temperature",
    "compute_time_constant",
    "create_adaptive_estimator",
    "create_single_room_estimator",
    "detect_anomalies_from_residuals",
    "detect_sustained_events",
    "estimate_from_multiple_segments",
    "estimate_parameters",
    "estimate_resistance_from_steady_state",
    "extract_parameters_from_ratios",
    "find_cooling_segments",
    "find_nighttime_cooling",
    "find_recurring_patterns",
    "fit_cooling_curve",
    "identify_thermal_bridges",
    "predict_building_temperature",
    "predict_temperature_change",
    "predict_temperature_trajectory",
    "rank_rooms_by_heat_loss",
    "separate_mass_and_resistance",
    "summarize_anomalies",
]
