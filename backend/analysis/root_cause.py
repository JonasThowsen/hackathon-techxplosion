"""Root cause analysis for waste patterns."""

from dataclasses import dataclass

from core.zones.base import Metrics, OverHeating, WastePattern


@dataclass
class RootCause:
    room_id: str
    pattern: str
    confidence: float  # 0.0 to 1.0
    explanation: str
    related_rooms: list[str]


def analyze_root_causes(
    room_metrics: dict[str, Metrics],
    waste_patterns: dict[str, list[WastePattern]],
    adjacency: dict[str, list[str]],
) -> list[RootCause]:
    """Cross-correlate waste patterns with room context for root cause analysis.

    Detections:
    - Temperature drop + high power -> open window
    - Temperature rise + low power + daytime -> solar gain
    - Multiple adjacent rooms with same anomaly -> building-level issue
    - Single room outlier vs peers -> room-specific issue
    """
    causes: list[RootCause] = []

    all_temps = [m.temperature for m in room_metrics.values()]
    avg_temp = sum(all_temps) / len(all_temps) if all_temps else 21.0

    for room_id, metrics in room_metrics.items():
        patterns = waste_patterns.get(room_id, [])
        neighbours = adjacency.get(room_id, [])
        has_overheating = any(isinstance(p, OverHeating) for p in patterns)

        hvac_power = metrics.total_hvac_power

        # Open window: temp below average while heating power is high
        if metrics.temperature < avg_temp - 2.0 and hvac_power > 150.0:
            confidence = 0.7
            if metrics.co2 < 420.0:
                confidence = 0.9  # low CO2 strongly suggests open window
            causes.append(
                RootCause(
                    room_id=room_id,
                    pattern="open_window",
                    confidence=confidence,
                    explanation=(
                        f"Temperature {metrics.temperature:.1f}C is well below "
                        f"average ({avg_temp:.1f}C) while HVAC power is "
                        f"{hvac_power:.0f}W — likely an open window"
                    ),
                    related_rooms=[],
                ),
            )

        # Solar gain: temp above average with low power
        if metrics.temperature > avg_temp + 2.0 and hvac_power < 80.0:
            causes.append(
                RootCause(
                    room_id=room_id,
                    pattern="solar_gain",
                    confidence=0.6,
                    explanation=(
                        f"Temperature {metrics.temperature:.1f}C is above "
                        f"average ({avg_temp:.1f}C) despite low HVAC power "
                        f"({hvac_power:.0f}W) — possible solar gain"
                    ),
                    related_rooms=[],
                ),
            )

        # Building-level insulation issue: same waste in adjacent rooms
        if has_overheating:
            affected_neighbours = [
                n for n in neighbours if any(isinstance(p, OverHeating) for p in waste_patterns.get(n, []))
            ]
            if len(affected_neighbours) >= 2:
                causes.append(
                    RootCause(
                        room_id=room_id,
                        pattern="insulation_issue",
                        confidence=0.75,
                        explanation=(
                            f"{room_id} and {len(affected_neighbours)} adjacent "
                            f"rooms are all overheating — possible building "
                            f"insulation or HVAC issue"
                        ),
                        related_rooms=affected_neighbours,
                    ),
                )

        # Single-room outlier: room is an outlier compared to neighbours
        if neighbours:
            neighbour_temps = [room_metrics[n].temperature for n in neighbours if n in room_metrics]
            if neighbour_temps:
                neighbour_avg = sum(neighbour_temps) / len(neighbour_temps)
                diff = abs(metrics.temperature - neighbour_avg)
                if diff > 3.0:
                    causes.append(
                        RootCause(
                            room_id=room_id,
                            pattern="room_specific_outlier",
                            confidence=min(0.5 + diff * 0.1, 0.95),
                            explanation=(
                                f"Temperature {metrics.temperature:.1f}C differs "
                                f"by {diff:.1f}C from neighbour average "
                                f"({neighbour_avg:.1f}C) — room-specific issue"
                            ),
                            related_rooms=neighbours,
                        ),
                    )

    return causes
