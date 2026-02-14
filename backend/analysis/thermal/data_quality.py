"""Data quality assessment for thermal parameter estimation.

Good parameter estimates require sufficient "excitation" in the data.
Without temperature variation, power changes, and dynamic conditions,
the parameters become unidentifiable (collinear in the regression).

This module provides tools to assess data quality before estimation
and warn about potential issues.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from analysis.thermal.types import RoomTimeSeries


class QualityLevel(Enum):
    """Data quality level."""

    EXCELLENT = "excellent"
    GOOD = "good"
    MARGINAL = "marginal"
    POOR = "poor"


@dataclass
class DataQualityMetrics:
    """Quantitative metrics for data quality assessment."""

    room_id: str

    # Temperature variation
    temp_range: float  # Max - min temperature (°C)
    temp_std: float  # Standard deviation of temperature
    temp_change_std: float  # Std of temperature changes (dynamics)

    # Power variation
    power_range: float  # Max - min power (W)
    power_std: float  # Standard deviation of power
    power_duty_cycle: float  # Fraction of time with power > threshold

    # Temporal coverage
    n_timesteps: int
    duration_hours: float

    # Excitation quality
    n_heating_cycles: int  # Number of on/off cycles
    has_cooling_periods: bool  # Whether heating is off for extended periods

    # Overall assessment
    quality_level: QualityLevel
    issues: list[str]


@dataclass
class QualityThresholds:
    """Thresholds for data quality assessment."""

    # Temperature thresholds
    min_temp_range: float = 2.0  # °C
    min_temp_std: float = 0.5  # °C
    min_temp_change_std: float = 0.05  # °C

    # Power thresholds
    min_power_range: float = 500.0  # W
    min_power_std: float = 200.0  # W
    min_duty_cycle: float = 0.2  # Fraction
    max_duty_cycle: float = 0.8  # Fraction

    # Temporal thresholds
    min_timesteps: int = 100
    min_hours: float = 24.0

    # Excitation thresholds
    min_heating_cycles: int = 5
    min_cooling_duration_timesteps: int = 10


def assess_room_data_quality(
    data: RoomTimeSeries,
    dt_seconds: float,
    thresholds: QualityThresholds | None = None,
) -> DataQualityMetrics:
    """Assess data quality for a single room.

    Args:
        data: Room time series
        dt_seconds: Timestep duration in seconds
        thresholds: Quality thresholds (uses defaults if None)

    Returns:
        Data quality metrics and assessment
    """
    if thresholds is None:
        thresholds = QualityThresholds()

    temps = np.array(data.temperature)
    powers = np.array(data.heating_power)
    n = len(temps)

    # Temperature metrics
    temp_range = float(np.max(temps) - np.min(temps))
    temp_std = float(np.std(temps))
    temp_changes = np.diff(temps)
    temp_change_std = float(np.std(temp_changes))

    # Power metrics
    power_range = float(np.max(powers) - np.min(powers))
    power_std = float(np.std(powers))
    power_on_threshold = 100.0  # W
    power_duty_cycle = float(np.mean(powers > power_on_threshold))

    # Temporal metrics
    duration_hours = n * dt_seconds / 3600.0

    # Heating cycles
    power_on = powers > power_on_threshold
    transitions = np.diff(power_on.astype(int))
    n_cycles = int(np.sum(np.abs(transitions)) // 2)

    # Cooling periods
    cooling_periods = _find_off_periods(powers, power_on_threshold)
    has_cooling = any(p[1] - p[0] >= thresholds.min_cooling_duration_timesteps for p in cooling_periods)

    # Identify issues
    issues: list[str] = []

    if temp_range < thresholds.min_temp_range:
        issues.append(f"Low temperature range: {temp_range:.2f}°C < {thresholds.min_temp_range}°C")

    if temp_std < thresholds.min_temp_std:
        issues.append(f"Low temperature variation: std={temp_std:.2f}°C")

    if temp_change_std < thresholds.min_temp_change_std:
        issues.append(f"Low temperature dynamics: dT_std={temp_change_std:.3f}°C")

    if power_range < thresholds.min_power_range:
        issues.append(f"Low power range: {power_range:.0f}W < {thresholds.min_power_range:.0f}W")

    if power_duty_cycle < thresholds.min_duty_cycle:
        issues.append(f"Heating rarely on: duty_cycle={power_duty_cycle:.1%}")

    if power_duty_cycle > thresholds.max_duty_cycle:
        issues.append(f"Heating always on: duty_cycle={power_duty_cycle:.1%}")

    if n < thresholds.min_timesteps:
        issues.append(f"Insufficient data: {n} < {thresholds.min_timesteps} timesteps")

    if duration_hours < thresholds.min_hours:
        issues.append(f"Short duration: {duration_hours:.1f}h < {thresholds.min_hours}h")

    if n_cycles < thresholds.min_heating_cycles:
        issues.append(f"Few heating cycles: {n_cycles} < {thresholds.min_heating_cycles}")

    if not has_cooling:
        issues.append("No extended cooling periods for thermal mass estimation")

    # Determine quality level
    if len(issues) == 0:
        quality = QualityLevel.EXCELLENT
    elif len(issues) <= 2:
        quality = QualityLevel.GOOD
    elif len(issues) <= 4:
        quality = QualityLevel.MARGINAL
    else:
        quality = QualityLevel.POOR

    return DataQualityMetrics(
        room_id=data.room_id,
        temp_range=temp_range,
        temp_std=temp_std,
        temp_change_std=temp_change_std,
        power_range=power_range,
        power_std=power_std,
        power_duty_cycle=power_duty_cycle,
        n_timesteps=n,
        duration_hours=duration_hours,
        n_heating_cycles=n_cycles,
        has_cooling_periods=has_cooling,
        quality_level=quality,
        issues=issues,
    )


def _find_off_periods(
    powers: NDArray[np.float64],
    threshold: float,
) -> list[tuple[int, int]]:
    """Find periods where power is below threshold."""
    off = powers < threshold
    periods: list[tuple[int, int]] = []

    in_period = False
    start = 0

    for i, is_off in enumerate(off):
        if is_off and not in_period:
            in_period = True
            start = i
        elif not is_off and in_period:
            in_period = False
            periods.append((start, i))

    if in_period:
        periods.append((start, len(off)))

    return periods


def assess_building_data_quality(
    room_data: dict[str, RoomTimeSeries],
    dt_seconds: float,
    thresholds: QualityThresholds | None = None,
) -> dict[str, DataQualityMetrics]:
    """Assess data quality for all rooms in a building.

    Args:
        room_data: Time series data for all rooms
        dt_seconds: Timestep duration
        thresholds: Quality thresholds

    Returns:
        Dict mapping room_id to quality metrics
    """
    return {room_id: assess_room_data_quality(data, dt_seconds, thresholds) for room_id, data in room_data.items()}


@dataclass
class EstimationReadiness:
    """Assessment of whether data is ready for estimation."""

    ready: bool
    quality_summary: str
    poor_rooms: list[str]
    recommendations: list[str]


def check_estimation_readiness(
    quality_metrics: dict[str, DataQualityMetrics],
) -> EstimationReadiness:
    """Check if data is ready for thermal parameter estimation.

    Args:
        quality_metrics: Quality metrics for all rooms

    Returns:
        Readiness assessment with recommendations
    """
    poor_rooms = [room_id for room_id, metrics in quality_metrics.items() if metrics.quality_level == QualityLevel.POOR]

    marginal_rooms = [
        room_id for room_id, metrics in quality_metrics.items() if metrics.quality_level == QualityLevel.MARGINAL
    ]

    good_rooms = [
        room_id
        for room_id, metrics in quality_metrics.items()
        if metrics.quality_level in (QualityLevel.GOOD, QualityLevel.EXCELLENT)
    ]

    # Determine readiness
    total = len(quality_metrics)
    ready = len(poor_rooms) == 0 and len(good_rooms) >= total * 0.5

    # Build summary
    quality_summary = (
        f"{len(good_rooms)} excellent/good, {len(marginal_rooms)} marginal, {len(poor_rooms)} poor out of {total} rooms"
    )

    # Build recommendations
    recommendations: list[str] = []

    if poor_rooms:
        recommendations.append(
            f"Rooms {poor_rooms} have poor data quality. Consider excluding from estimation or collecting more data."
        )

    # Check for common issues across rooms
    all_issues: list[str] = []
    for metrics in quality_metrics.values():
        all_issues.extend(metrics.issues)

    issue_counts: dict[str, int] = {}
    for issue in all_issues:
        # Extract issue type (first few words)
        issue_type = issue.split(":")[0] if ":" in issue else issue
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        if count > len(quality_metrics) * 0.5:
            recommendations.append(
                f"Common issue across {count} rooms: {issue_type}. Consider adjusting data collection."
            )

    if not recommendations:
        recommendations.append("Data quality is sufficient for estimation.")

    return EstimationReadiness(
        ready=ready,
        quality_summary=quality_summary,
        poor_rooms=poor_rooms,
        recommendations=recommendations,
    )


def compute_signal_to_noise(
    data: RoomTimeSeries,
    model_rmse: float,
) -> float:
    """Compute signal-to-noise ratio for temperature changes.

    Higher SNR means better parameter estimates.

    Args:
        data: Room time series
        model_rmse: RMSE from model fit

    Returns:
        Signal-to-noise ratio (dimensionless)
    """
    temps = np.array(data.temperature)
    temp_changes = np.diff(temps)
    signal_std = float(np.std(temp_changes))

    if model_rmse == 0:
        return float("inf")

    return signal_std / model_rmse


def check_collinearity(
    room_data: dict[str, RoomTimeSeries],
    external_temps: list[float],
) -> dict[str, float]:
    """Check for collinearity between temperature variables.

    High collinearity between room temperatures and external temperature
    can make parameters difficult to distinguish.

    Args:
        room_data: Time series for all rooms
        external_temps: External temperature series

    Returns:
        Dict mapping room_id to correlation with external temperature
    """
    ext = np.array(external_temps)
    correlations: dict[str, float] = {}

    for room_id, data in room_data.items():
        temps = np.array(data.temperature[: len(ext)])
        if len(temps) > 0 and np.std(temps) > 0 and np.std(ext) > 0:
            corr = float(np.corrcoef(temps, ext[: len(temps)])[0, 1])
            correlations[room_id] = corr
        else:
            correlations[room_id] = 0.0

    return correlations
