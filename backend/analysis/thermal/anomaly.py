"""Thermal anomaly detection (window/door openings, equipment faults).

This module provides two methods for detecting thermal anomalies:

Method A - Residual Monitoring:
    Predict temperatures using the calibrated model, then flag timesteps
    where the residual (observed - predicted) exceeds a threshold.
    Works with batch estimation results.

Method B - Time-varying Parameter Estimation:
    Use RLS with forgetting factor to track parameters over time.
    Sudden changes in estimated conductance indicate anomalies.
    Better for real-time detection.
"""

from dataclasses import dataclass

import numpy as np

from analysis.thermal.types import (
    AnomalyDetectionConfig,
    AnomalyReport,
    ThermalAnomaly,
    TimestepResidual,
)

# -----------------------------------------------------------------------------
# Residual-Based Detection (Method A)
# -----------------------------------------------------------------------------


def detect_anomalies_from_residuals(
    residuals: list[TimestepResidual],
    config: AnomalyDetectionConfig | None = None,
) -> AnomalyReport:
    """Detect anomalies from estimation residuals.

    Large positive residuals indicate unexpected heat loss (open window/door).
    Large negative residuals indicate unexpected heat gain.

    Args:
        residuals: Per-timestep residuals from parameter estimation
        config: Detection configuration

    Returns:
        Report of detected anomalies
    """
    if config is None:
        config = AnomalyDetectionConfig()

    # Group residuals by room
    by_room: dict[str, list[TimestepResidual]] = {}
    for r in residuals:
        if r.room_id not in by_room:
            by_room[r.room_id] = []
        by_room[r.room_id].append(r)

    anomalies: list[ThermalAnomaly] = []
    anomalous_timesteps: dict[str, list[int]] = {}

    for room_id, room_residuals in by_room.items():
        # Sort by timestep
        room_residuals.sort(key=lambda r: r.timestep)

        # Compute statistics
        resid_values = np.array([r.residual for r in room_residuals])
        mean_r = float(np.mean(resid_values))
        std_r = float(np.std(resid_values))

        if std_r == 0:
            anomalous_timesteps[room_id] = []
            continue

        threshold = config.threshold_std * std_r

        # Find anomalous timesteps
        flagged: list[int] = []
        for r in room_residuals:
            if abs(r.residual - mean_r) > threshold:
                flagged.append(r.timestep)

        anomalous_timesteps[room_id] = flagged

        # Group consecutive anomalies into events
        events = _group_consecutive_timesteps(flagged, config.min_consecutive)

        for start, end in events:
            # Compute event characteristics
            event_residuals = [r for r in room_residuals if start <= r.timestep <= end]
            avg_residual = float(np.mean([r.residual for r in event_residuals]))

            # Determine anomaly type
            anomaly_type = "heat_loss" if avg_residual > 0 else "heat_gain"

            # Estimate extra conductance from average residual
            # residual ≈ (G_extra/C) * dt * (T - T_ext)
            # This is approximate without knowing all the values
            extra_g = _estimate_extra_conductance(event_residuals)

            anomalies.append(
                ThermalAnomaly(
                    room_id=room_id,
                    start_timestep=start,
                    end_timestep=end,
                    anomaly_type=anomaly_type,
                    magnitude=abs(avg_residual),
                    estimated_extra_conductance=extra_g,
                )
            )

    return AnomalyReport(
        anomalies=anomalies,
        anomalous_timesteps_by_room=anomalous_timesteps,
    )


def _group_consecutive_timesteps(
    timesteps: list[int],
    min_consecutive: int,
) -> list[tuple[int, int]]:
    """Group consecutive timesteps into (start, end) ranges.

    Only returns groups with at least min_consecutive elements.
    """
    if not timesteps:
        return []

    groups: list[tuple[int, int]] = []
    start = timesteps[0]
    prev = timesteps[0]

    for t in timesteps[1:]:
        if t == prev + 1:
            prev = t
        else:
            if prev - start + 1 >= min_consecutive:
                groups.append((start, prev))
            start = t
            prev = t

    # Don't forget the last group
    if prev - start + 1 >= min_consecutive:
        groups.append((start, prev))

    return groups


def _estimate_extra_conductance(
    residuals: list[TimestepResidual],
) -> float:
    """Estimate extra conductance causing the anomaly.

    This is a rough estimate based on the average residual magnitude
    and typical temperature differences.
    """
    if not residuals:
        return 0.0

    avg_residual = float(np.mean([abs(r.residual) for r in residuals]))

    # Assume:
    # - dt = 180s (typical)
    # - C = 500 kJ/K (typical room)
    # - ΔT = 15 K (typical indoor-outdoor difference)
    # Then: residual ≈ (G_extra * dt * ΔT) / C
    # So: G_extra ≈ residual * C / (dt * ΔT)

    dt_assumed = 180.0
    C_assumed = 5e5
    delta_T_assumed = 15.0

    extra_g = avg_residual * C_assumed / (dt_assumed * delta_T_assumed)
    return abs(extra_g)


# -----------------------------------------------------------------------------
# Rolling Window Detection
# -----------------------------------------------------------------------------


@dataclass
class RollingAnomalyDetector:
    """Online anomaly detector using rolling statistics."""

    window_size: int = 20
    threshold_std: float = 3.0
    min_samples: int = 10

    def __post_init__(self) -> None:
        self._buffers: dict[str, list[float]] = {}

    def update(
        self,
        room_id: str,
        residual: float,
    ) -> bool:
        """Update detector with new residual and check for anomaly.

        Args:
            room_id: Room identifier
            residual: New residual value

        Returns:
            True if this timestep is anomalous
        """
        if room_id not in self._buffers:
            self._buffers[room_id] = []

        buffer = self._buffers[room_id]
        buffer.append(residual)

        # Keep window size
        if len(buffer) > self.window_size:
            buffer.pop(0)

        # Need minimum samples
        if len(buffer) < self.min_samples:
            return False

        # Compute statistics on all but last sample
        history = buffer[:-1]
        mean = float(np.mean(history))
        std = float(np.std(history))

        if std == 0:
            return False

        return abs(residual - mean) > self.threshold_std * std


# -----------------------------------------------------------------------------
# Sustained Event Detection
# -----------------------------------------------------------------------------


def detect_sustained_events(
    anomalies: list[ThermalAnomaly],
    min_duration_timesteps: int = 4,
) -> list[ThermalAnomaly]:
    """Filter anomalies to only include sustained events.

    Short spikes may be measurement noise. True events like open
    windows typically last multiple timesteps.

    Args:
        anomalies: All detected anomalies
        min_duration_timesteps: Minimum duration to consider sustained

    Returns:
        Filtered list of sustained anomalies
    """
    return [a for a in anomalies if (a.end_timestep - a.start_timestep + 1) >= min_duration_timesteps]


def classify_anomaly_severity(
    anomaly: ThermalAnomaly,
    mild_threshold: float = 0.1,
    moderate_threshold: float = 0.3,
) -> str:
    """Classify anomaly severity based on magnitude.

    Args:
        anomaly: The anomaly to classify
        mild_threshold: Below this is 'mild'
        moderate_threshold: Above this is 'severe'

    Returns:
        Severity string: 'mild', 'moderate', or 'severe'
    """
    if anomaly.magnitude < mild_threshold:
        return "mild"
    elif anomaly.magnitude < moderate_threshold:
        return "moderate"
    else:
        return "severe"


# -----------------------------------------------------------------------------
# Pattern Analysis
# -----------------------------------------------------------------------------


def find_recurring_patterns(
    anomalies: list[ThermalAnomaly],
    timesteps_per_day: int = 96,  # 15-minute intervals
) -> dict[str, list[int]]:
    """Find recurring anomaly patterns (e.g., same time each day).

    Args:
        anomalies: Detected anomalies
        timesteps_per_day: Number of timesteps in a day

    Returns:
        Dict mapping room_id to list of common hours when anomalies occur
    """
    # Count anomalies by hour of day for each room
    hour_counts: dict[str, dict[int, int]] = {}

    timesteps_per_hour = timesteps_per_day // 24

    for anomaly in anomalies:
        room_id = anomaly.room_id
        if room_id not in hour_counts:
            hour_counts[room_id] = dict.fromkeys(range(24), 0)

        start_hour = (anomaly.start_timestep % timesteps_per_day) // timesteps_per_hour
        hour_counts[room_id][start_hour] += 1

    # Find hours with above-average anomaly counts
    patterns: dict[str, list[int]] = {}
    for room_id, counts in hour_counts.items():
        total = sum(counts.values())
        if total == 0:
            patterns[room_id] = []
            continue

        avg = total / 24
        frequent_hours = [h for h, c in counts.items() if c > avg * 2]
        patterns[room_id] = sorted(frequent_hours)

    return patterns


def summarize_anomalies(
    report: AnomalyReport,
) -> dict[str, dict[str, int | float]]:
    """Generate summary statistics for anomalies.

    Args:
        report: Anomaly detection report

    Returns:
        Summary dict with counts and totals per room
    """
    summary: dict[str, dict[str, int | float]] = {}

    # Group by room
    by_room: dict[str, list[ThermalAnomaly]] = {}
    for anomaly in report.anomalies:
        if anomaly.room_id not in by_room:
            by_room[anomaly.room_id] = []
        by_room[anomaly.room_id].append(anomaly)

    for room_id, anomalies in by_room.items():
        n_events = len(anomalies)
        total_duration = sum(a.end_timestep - a.start_timestep + 1 for a in anomalies)
        avg_magnitude = float(np.mean([a.magnitude for a in anomalies])) if anomalies else 0.0
        heat_loss_events = len([a for a in anomalies if a.anomaly_type == "heat_loss"])
        heat_gain_events = len([a for a in anomalies if a.anomaly_type == "heat_gain"])

        summary[room_id] = {
            "n_events": n_events,
            "total_duration_timesteps": total_duration,
            "avg_magnitude": avg_magnitude,
            "heat_loss_events": heat_loss_events,
            "heat_gain_events": heat_gain_events,
        }

    return summary
