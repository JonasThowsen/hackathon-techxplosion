"""Cooling curve analysis for separating thermal mass and conductivity.

The cooling curve method uses nighttime or heating-off periods to
separate thermal mass (C) from thermal resistance (R).

During free cooling (no heating, constant neighbors):
    dT/dt = -(1/RC) * (T - T_ambient)

Solution:
    T(t) = T_ambient + (T_0 - T_ambient) * exp(-t/τ)

Where τ = R*C is the time constant.

If we know R from steady-state analysis:
    R = ΔT / P_steady

Then we can compute:
    C = τ / R

This provides the best separation of these otherwise collinear parameters.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit  # pyright: ignore[reportUnknownVariableType]

from analysis.thermal.types import (
    CoolingCurveResult,
    CoolingCurveSegment,
    RoomTimeSeries,
)

# -----------------------------------------------------------------------------
# Segment Detection
# -----------------------------------------------------------------------------


def find_cooling_segments(
    data: RoomTimeSeries,
    external_temps: list[float],
    power_threshold: float = 50.0,
    min_duration_timesteps: int = 10,
    min_temp_drop: float = 0.5,
) -> list[CoolingCurveSegment]:
    """Find segments where the room is cooling naturally (heating off).

    Good cooling segments have:
    - Heating power near zero
    - Monotonic or near-monotonic temperature decrease
    - Sufficient duration for curve fitting

    Args:
        data: Room time series data
        external_temps: External temperature series
        power_threshold: Max heating power to consider "off"
        min_duration_timesteps: Minimum segment length
        min_temp_drop: Minimum temperature drop during segment

    Returns:
        List of cooling curve segments
    """
    n = len(data)
    segments: list[CoolingCurveSegment] = []

    i = 0
    while i < n:
        # Find start of cooling period (power off, temp above external)
        if data.heating_power[i] <= power_threshold:
            start = i

            # Extend until heating comes back on
            while i < n and data.heating_power[i] <= power_threshold:
                i += 1
            end = i - 1

            # Check segment quality
            if end - start + 1 >= min_duration_timesteps:
                start_temp = data.temperature[start]
                end_temp = data.temperature[end]
                temp_drop = start_temp - end_temp

                if temp_drop >= min_temp_drop:
                    avg_external = float(np.mean(external_temps[start : end + 1]))

                    segments.append(
                        CoolingCurveSegment(
                            room_id=data.room_id,
                            start_timestep=start,
                            end_timestep=end,
                            start_temp=start_temp,
                            end_temp=end_temp,
                            avg_external_temp=avg_external,
                        )
                    )
        else:
            i += 1

    return segments


def find_nighttime_cooling(
    data: RoomTimeSeries,
    external_temps: list[float],
    dt_seconds: float,
    night_start_hour: int = 22,
    night_end_hour: int = 6,
) -> list[CoolingCurveSegment]:
    """Find nighttime cooling segments.

    Nighttime cooling is particularly useful because:
    - Heating is often in setback mode
    - Solar gains are zero
    - Occupancy loads are minimal

    Args:
        data: Room time series data
        external_temps: External temperature series
        dt_seconds: Timestep duration in seconds
        night_start_hour: Hour when night period starts (0-23)
        night_end_hour: Hour when night period ends (0-23)

    Returns:
        List of nighttime cooling segments
    """
    segments: list[CoolingCurveSegment] = []
    timesteps_per_hour = 3600.0 / dt_seconds

    i = 0
    while i < len(data):
        # Check if this timestep is during night hours
        hour = (i / timesteps_per_hour) % 24

        is_night = (night_start_hour <= hour) or (hour < night_end_hour)

        if is_night and data.heating_power[i] < 100:  # Minimal heating
            start = i

            # Extend through night
            while i < len(data):
                hour = (i / timesteps_per_hour) % 24
                is_still_night = (night_start_hour <= hour) or (hour < night_end_hour)
                if not is_still_night or data.heating_power[i] >= 100:
                    break
                i += 1

            end = i - 1

            if end - start >= 10:  # At least 10 timesteps
                segments.append(
                    CoolingCurveSegment(
                        room_id=data.room_id,
                        start_timestep=start,
                        end_timestep=end,
                        start_temp=data.temperature[start],
                        end_temp=data.temperature[end],
                        avg_external_temp=float(np.mean(external_temps[start : end + 1])),
                    )
                )
        else:
            i += 1

    return segments


# -----------------------------------------------------------------------------
# Curve Fitting
# -----------------------------------------------------------------------------


def _exponential_decay(
    t: NDArray[np.float64],
    t_ambient: float,
    delta_t0: float,
    tau: float,
) -> NDArray[np.float64]:
    """Exponential decay model for cooling.

    T(t) = T_ambient + ΔT_0 * exp(-t/τ)

    Args:
        t: Time array (seconds from start)
        t_ambient: Ambient (equilibrium) temperature
        delta_t0: Initial temperature difference (T_0 - T_ambient)
        tau: Time constant (seconds)

    Returns:
        Temperature array
    """
    return t_ambient + delta_t0 * np.exp(-t / tau)


def fit_cooling_curve(
    segment: CoolingCurveSegment,
    temperatures: list[float],
    dt_seconds: float,
) -> CoolingCurveResult:
    """Fit exponential cooling curve to a segment.

    Args:
        segment: The cooling segment to fit
        temperatures: Full temperature time series
        dt_seconds: Timestep duration in seconds

    Returns:
        Fitted cooling curve result
    """
    # Extract segment data
    start = segment.start_timestep
    end = segment.end_timestep
    temps = np.array(temperatures[start : end + 1], dtype=np.float64)
    n = len(temps)

    # Time array (seconds from start)
    t = np.arange(n, dtype=np.float64) * dt_seconds

    # Initial guesses
    t_ambient_guess = segment.avg_external_temp
    delta_t0_guess = temps[0] - t_ambient_guess
    tau_guess = (end - start) * dt_seconds / 2  # Half the segment duration

    try:
        # Fit the curve
        popt, _ = curve_fit(  # pyright: ignore[reportUnknownVariableType]
            _exponential_decay,
            t,
            temps,
            p0=[t_ambient_guess, delta_t0_guess, tau_guess],
            bounds=(
                [segment.avg_external_temp - 5, 0, 60],  # Lower bounds
                [segment.avg_external_temp + 5, 50, 1e6],  # Upper bounds
            ),
            maxfev=5000,
        )

        t_ambient_fit = float(popt[0])  # pyright: ignore[reportUnknownArgumentType]
        delta_t0_fit = float(popt[1])  # pyright: ignore[reportUnknownArgumentType]
        tau_fit = float(popt[2])  # pyright: ignore[reportUnknownArgumentType]

        # Compute R-squared
        temps_pred = _exponential_decay(t, t_ambient_fit, delta_t0_fit, tau_fit)
        ss_res = float(np.sum((temps - temps_pred) ** 2))
        ss_tot = float(np.sum((temps - np.mean(temps)) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return CoolingCurveResult(
            room_id=segment.room_id,
            time_constant_seconds=float(tau_fit),
            estimated_resistance=None,  # Need steady state to compute
            estimated_capacitance=None,  # Need R to compute
            fit_r_squared=r_squared,
        )

    except RuntimeError:
        # Curve fitting failed
        return CoolingCurveResult(
            room_id=segment.room_id,
            time_constant_seconds=float("nan"),
            estimated_resistance=None,
            estimated_capacitance=None,
            fit_r_squared=0.0,
        )


# -----------------------------------------------------------------------------
# Parameter Separation
# -----------------------------------------------------------------------------


def separate_mass_and_resistance(
    cooling_result: CoolingCurveResult,
    steady_state_power: float,
    steady_state_temp_diff: float,
) -> tuple[float, float]:
    """Separate thermal mass and resistance using cooling curve + steady state.

    At steady state:
        R = ΔT / P

    From cooling curve:
        τ = R * C

    Therefore:
        C = τ / R = τ * P / ΔT

    Args:
        cooling_result: Result from cooling curve fitting
        steady_state_power: Heating power at steady state (W)
        steady_state_temp_diff: Indoor - outdoor temperature difference (K)

    Returns:
        Tuple of (thermal_mass_J_K, thermal_resistance_K_W)
    """
    tau = cooling_result.time_constant_seconds

    if np.isnan(tau) or steady_state_power == 0:
        raise ValueError("Cannot separate parameters: invalid inputs")

    # Thermal resistance from steady state
    R = abs(steady_state_temp_diff) / steady_state_power

    # Thermal capacitance from time constant
    C = tau / R

    return C, R


def estimate_from_multiple_segments(
    segments: list[CoolingCurveSegment],
    temperatures: list[float],
    dt_seconds: float,
) -> tuple[float, float]:
    """Estimate average time constant from multiple cooling segments.

    Using multiple segments provides:
    - Better statistical confidence
    - Detection of outliers (e.g., segments with open windows)

    Args:
        segments: List of cooling segments
        temperatures: Full temperature series
        dt_seconds: Timestep duration

    Returns:
        Tuple of (mean_tau, std_tau) in seconds
    """
    tau_values: list[float] = []

    for segment in segments:
        result = fit_cooling_curve(segment, temperatures, dt_seconds)
        if not np.isnan(result.time_constant_seconds) and result.fit_r_squared > 0.8:
            tau_values.append(result.time_constant_seconds)

    if not tau_values:
        raise ValueError("No valid cooling curves could be fitted")

    mean_tau = float(np.mean(tau_values))
    std_tau = float(np.std(tau_values))

    return mean_tau, std_tau


# -----------------------------------------------------------------------------
# Identifiability Analysis
# -----------------------------------------------------------------------------


@dataclass
class IdentifiabilityReport:
    """Report on parameter identifiability from available data."""

    has_cooling_data: bool
    has_steady_state_data: bool
    can_separate_parameters: bool
    n_cooling_segments: int
    recommended_approach: str


def analyze_identifiability(
    data: RoomTimeSeries,
    external_temps: list[float],
) -> IdentifiabilityReport:
    """Analyze whether thermal mass and resistance can be separated.

    For full identifiability, we need:
    1. Cooling curves (heating off, temperature decreasing)
    2. Steady-state periods (constant temperature, known power)

    Without both, we can only estimate the ratio C/G or the product RC.

    Args:
        data: Room time series
        external_temps: External temperature series

    Returns:
        Identifiability report with recommendations
    """
    # Find cooling segments
    cooling_segments = find_cooling_segments(data, external_temps)
    has_cooling = len(cooling_segments) > 0

    # Check for steady state periods
    # Steady state: low temperature variance over extended period
    temp_array = np.array(data.temperature)
    power_array = np.array(data.heating_power)

    # Look for periods with stable temperature and non-zero power
    window = 20  # timesteps
    has_steady_state = False

    for i in range(len(temp_array) - window):
        temp_window = temp_array[i : i + window]
        power_window = power_array[i : i + window]

        temp_std = float(np.std(temp_window))
        avg_power = float(np.mean(power_window))

        if temp_std < 0.2 and avg_power > 100:
            has_steady_state = True
            break

    can_separate = has_cooling and has_steady_state

    if can_separate:
        approach = "Use cooling curve fitting + steady state to fully identify C and R"
    elif has_cooling:
        approach = (
            "Can estimate time constant τ=RC from cooling curves, but cannot separate C and R without steady state data"
        )
    elif has_steady_state:
        approach = "Can estimate R from steady state, but cannot separate C without cooling data"
    else:
        approach = "Insufficient data for parameter separation. Only ratios like G/C can be estimated."

    return IdentifiabilityReport(
        has_cooling_data=has_cooling,
        has_steady_state_data=has_steady_state,
        can_separate_parameters=can_separate,
        n_cooling_segments=len(cooling_segments),
        recommended_approach=approach,
    )
