"""Prediction service for temperature forecasting.

This service provides real-time temperature predictions using a hybrid approach:
1. First-principles (geometry-based) parameters for immediate predictions
2. Estimated parameters from historical data when available

Predictions are used for:
- Pre-emptive heating control
- Display in frontend with cost estimates
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict

import numpy as np

from analysis.thermal import (
    ExternalConditions,
    RoomTimeSeries,
    ThermalGraph,
    ThermalParameters,
    build_graph_from_adjacency,
    create_parameters_from_geometry,
    estimate_parameters,
    predict_building_temperature,
)
from analysis.thermal.rc_model import RoomGeometry
from core.models import BuildingLayout


class PredictionEntry(TypedDict):
    room_id: str
    predicted_temps: list[float]
    uncertainty: float
    warnings: list[str]
    will_be_cold: bool
    will_be_hot: bool


@dataclass
class RoomPrediction:
    """Prediction result for a single room."""

    room_id: str
    predicted_temps: list[float]
    uncertainty: float
    warnings: list[str]
    will_be_cold: bool
    will_be_hot: bool
    uses_estimated_params: bool = False


@dataclass
class PredictionResult:
    """Prediction results for all rooms."""

    predictions: dict[str, RoomPrediction]
    horizon_ticks: int
    tick_interval_minutes: float
    generated_at: datetime = field(default_factory=datetime.now)


PREDICTION_HORIZON_TICKS = 40  # 2 hours at 3 min/tick
PREDICTION_UPDATE_INTERVAL_TICKS = 5  # Update every 5 ticks (15 min)
THERMAL_TIME_CONSTANT_HOURS = 4.0  # Expected thermal time constant


class PredictionService:
    """Service for predicting room temperatures."""

    _geometry_params: ThermalParameters | None = None
    _estimated_params: ThermalParameters | None = None
    _thermal_graph: ThermalGraph | None = None
    _last_update_tick: int = -999
    _last_estimation_rmse: float = 999.0
    _room_geometries: dict[str, RoomGeometry] = {}  # type: ignore[assignment, valid-type]

    @classmethod
    def initialize_from_building(cls, building: BuildingLayout) -> None:
        """Initialize prediction service from building layout."""
        rooms: list[RoomGeometry] = []

        for floor in building.floors:
            for room in floor.rooms:
                polygon = room.polygon
                if len(polygon) < 3:
                    continue

                min_x = min(p[0] for p in polygon)
                max_x = max(p[0] for p in polygon)
                min_y = min(p[1] for p in polygon)
                max_y = max(p[1] for p in polygon)

                floor_area = (max_x - min_x) * (max_y - min_y)
                wall_perim = 2 * ((max_x - min_x) + (max_y - min_y))
                wall_area = wall_perim * 3.0
                window_area = wall_area * 0.2

                is_exterior = floor.floor_index == 0

                neighbor_ids = []
                for other_floor in building.floors:
                    for other_room in other_floor.rooms:
                        if other_room.id != room.id:
                            other_polygon = other_room.polygon
                            if len(other_polygon) >= 3 and cls._rooms_adjacent(polygon, other_polygon):
                                neighbor_ids.append(other_room.id)

                geom = RoomGeometry(
                    room_id=room.id,
                    floor_area_m2=floor_area,
                    wall_area_m2=wall_area,
                    window_area_m2=window_area,
                    ceiling_height_m=3.0,
                    is_exterior=is_exterior,
                    neighbor_ids=neighbor_ids,
                )
                cls._room_geometries[room.id] = geom
                rooms.append(geom)

        cls._geometry_params = create_parameters_from_geometry(rooms)

        room_ids = [r.id for floor in building.floors for r in floor.rooms]
        adjacency: dict[str, list[str]] = {rid: [] for rid in room_ids}
        for geom in rooms:
            for nid in geom.neighbor_ids:
                if nid in adjacency:
                    adjacency[geom.room_id].append(nid)

        exterior_rooms: set[str] = set()
        if building.floors:
            exterior_rooms = {r.id for r in building.floors[0].rooms}

        cls._thermal_graph = build_graph_from_adjacency(room_ids, adjacency, None, exterior_rooms)  # type: ignore[arg-type]

    @staticmethod
    def _rooms_adjacent(poly1: list[list[float]], poly2: list[list[float]]) -> bool:
        """Check if two room polygons are adjacent."""
        min_x1, max_x1 = min(p[0] for p in poly1), max(p[0] for p in poly1)
        min_y1, max_y1 = min(p[1] for p in poly1), max(p[1] for p in poly1)
        min_x2, max_x2 = min(p[0] for p in poly2), max(p[0] for p in poly2)
        min_y2, max_y2 = min(p[1] for p in poly2), max(p[1] for p in poly2)

        return (
            (abs(max_x1 - min_x2) < 0.5 and max_y1 > min_y2 and min_y1 < max_y2)
            or (abs(max_x2 - min_x1) < 0.5 and max_y2 > min_y1 and min_y2 < max_y1)
            or (abs(max_y1 - min_y2) < 0.5 and max_x1 > min_x2 and min_x1 < max_x2)
            or (abs(max_y2 - min_y1) < 0.5 and max_x2 > min_x1 and min_x2 < max_x1)
        )

    @classmethod
    def update_estimation(
        cls,
        room_data: dict[str, RoomTimeSeries],
        external_temp: float,
    ) -> float:
        """Update estimated thermal parameters from historical data.

        Returns:
            RMSE of the estimation fit, or 999.0 if estimation fails
        """
        if cls._thermal_graph is None:
            return 999.0

        try:
            result = estimate_parameters(
                room_data=room_data,
                graph=cls._thermal_graph,
                external=ExternalConditions(temperature=external_temp),
            )

            if result.fit_metrics is not None:
                cls._estimated_params = result.parameters
                cls._last_estimation_rmse = result.fit_metrics.rmse
                return result.fit_metrics.rmse

        except Exception:
            pass

        cls._last_estimation_rmse = 999.0
        return 999.0

    @classmethod
    def predict(
        cls,
        current_temps: dict[str, float],
        heating_powers: dict[str, float],
        external_temp: float,
        neighbor_heat_inflow: dict[str, float],
        tick: int,
    ) -> PredictionResult:
        """Generate temperature predictions for all rooms.

        Args:
            current_temps: Current temperature of each room (°C)
            heating_powers: Current heating power of each room (W)
            external_temp: Current external temperature (°C)
            neighbor_heat_inflow: Heat inflow from neighbors per room (W)
            tick: Current simulation tick

        Returns:
            PredictionResult with predictions for all rooms
        """
        if cls._thermal_graph is None or cls._geometry_params is None:
            return PredictionResult(predictions={}, horizon_ticks=PREDICTION_HORIZON_TICKS, tick_interval_minutes=3.0)

        params = cls._estimated_params if cls._estimated_params is not None else cls._geometry_params
        uses_estimated = cls._estimated_params is not None

        horizon = PREDICTION_HORIZON_TICKS
        dt_seconds = 180.0
        tick_interval_minutes = dt_seconds / 60.0

        room_ids = list(current_temps.keys())
        initial_temps = current_temps.copy()

        heating_schedule: dict[str, np.ndarray] = {}
        for room_id in room_ids:
            current_power = heating_powers.get(room_id, 0.0)
            power_estimate = current_power + neighbor_heat_inflow.get(room_id, 0.0)
            base_power = max(0.0, power_estimate)

            powers = np.full(horizon, base_power, dtype=np.float64)

            natural_cooling = 0.3
            for i in range(1, min(10, horizon)):
                powers[i] = powers[i] * (1 - i * natural_cooling / horizon)

            heating_schedule[room_id] = powers

        external_temps = np.full(horizon, external_temp, dtype=np.float64)

        try:
            prediction = predict_building_temperature(
                parameters=params,
                graph=cls._thermal_graph,
                initial_temps=initial_temps,
                heating_schedule=heating_schedule,
                external_temps=external_temps,
                dt_seconds=dt_seconds,
            )
        except Exception:
            return PredictionResult(predictions={}, horizon_ticks=horizon, tick_interval_minutes=tick_interval_minutes)

        baseline_temp = 21.0
        cold_threshold = baseline_temp - 0.5
        hot_threshold = baseline_temp + 1.0

        uncertainty = cls._last_estimation_rmse if uses_estimated else 2.0
        warnings: list[str] = []
        if uncertainty > 2.0:
            warnings.append("high_uncertainty")
        if not uses_estimated:
            warnings.append("using_geometry_params")

        predictions: dict[str, RoomPrediction] = {}

        for room_id in room_ids:
            if room_id not in prediction.room_temps:
                continue

            temps = prediction.room_temps[room_id]

            idx_30min = min(10, horizon - 1)
            idx_1h = min(20, horizon - 1)
            idx_2h = min(40, horizon - 1)

            predicted_30min = float(temps[idx_30min]) if len(temps) > idx_30min else current_temps.get(room_id, 20.0)
            predicted_1h = float(temps[idx_1h]) if len(temps) > idx_1h else current_temps.get(room_id, 20.0)
            predicted_2h = float(temps[idx_2h]) if len(temps) > idx_2h else current_temps.get(room_id, 20.0)

            will_be_cold = predicted_1h < cold_threshold or predicted_2h < cold_threshold
            will_be_hot = predicted_1h > hot_threshold or predicted_2h > hot_threshold

            predictions[room_id] = RoomPrediction(
                room_id=room_id,
                predicted_temps=[predicted_30min, predicted_1h, predicted_2h],
                uncertainty=uncertainty,
                warnings=warnings.copy(),
                will_be_cold=will_be_cold,
                will_be_hot=will_be_hot,
                uses_estimated_params=uses_estimated,
            )

        return PredictionResult(
            predictions=predictions,
            horizon_ticks=horizon,
            tick_interval_minutes=tick_interval_minutes,
        )

    @classmethod
    def should_update(cls, tick: int) -> bool:
        """Check if predictions should be updated at this tick."""
        return tick - cls._last_update_tick >= PREDICTION_UPDATE_INTERVAL_TICKS

    @classmethod
    def mark_updated(cls, tick: int) -> None:
        """Mark predictions as updated."""
        cls._last_update_tick = tick
