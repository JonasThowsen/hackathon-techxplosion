"""Analysis utilities for the EnergyZone hierarchy.

Pure functions that operate on metrics data -- no EnergyZone subclassing.
Provides room adjacency detection, heat flow calculation, and root cause analysis.
"""

from dataclasses import dataclass

from energy_zone import Metrics, WastePattern
from models import Room

# Two vertices are "the same" if within this distance (metres).
_EPSILON = 0.01

# Interior wall U-value in W/(m^2*K)
_U_INTERIOR = 1.5

# Assumed floor-to-ceiling height in metres
_WALL_HEIGHT_M = 3.0


# ---------------------------------------------------------------------------
# 1. Room adjacency
# ---------------------------------------------------------------------------


def find_adjacent_rooms(rooms: list[Room]) -> dict[str, list[str]]:
    """Determine which rooms share walls based on polygon geometry.

    Two rooms are adjacent if they share at least two polygon vertices
    (within a small epsilon). Works well for axis-aligned rectangles.
    """
    adjacency: dict[str, list[str]] = {r.id: [] for r in rooms}

    for i, a in enumerate(rooms):
        verts_a = {(_snap(p[0]), _snap(p[1])) for p in a.polygon}
        for b in rooms[i + 1 :]:
            verts_b = {(_snap(p[0]), _snap(p[1])) for p in b.polygon}
            shared = verts_a & verts_b
            if len(shared) >= 2:
                adjacency[a.id].append(b.id)
                adjacency[b.id].append(a.id)

    return adjacency


def _snap(v: float) -> float:
    """Snap a coordinate to the nearest epsilon grid."""
    return round(v / _EPSILON) * _EPSILON


# ---------------------------------------------------------------------------
# 2. Heat flow
# ---------------------------------------------------------------------------


@dataclass
class HeatFlow:
    from_room: str
    to_room: str
    flow_watts: float  # positive = heat flowing from -> to
    shared_wall_length_m: float


def calculate_heat_flows(
    room_metrics: dict[str, Metrics],
    adjacency: dict[str, list[str]],
    rooms: list[Room],
) -> list[HeatFlow]:
    """Calculate heat flow between adjacent rooms.

    Q = U * A * dT where:
    - U = 1.5 W/(m^2*K) for interior walls
    - A = shared wall length * wall height (3 m)
    - dT = temperature difference between rooms
    """
    room_polys: dict[str, list[list[float]]] = {r.id: r.polygon for r in rooms}
    seen: set[tuple[str, str]] = set()
    flows: list[HeatFlow] = []

    for room_id, neighbours in adjacency.items():
        for neighbour_id in neighbours:
            pair = (min(room_id, neighbour_id), max(room_id, neighbour_id))
            if pair in seen:
                continue
            seen.add(pair)

            wall_len = _shared_wall_length(room_polys[room_id], room_polys[neighbour_id])
            if wall_len < _EPSILON:
                continue

            m_a = room_metrics.get(room_id)
            m_b = room_metrics.get(neighbour_id)
            if m_a is None or m_b is None:
                continue

            dt = m_a.temperature - m_b.temperature
            area = wall_len * _WALL_HEIGHT_M
            q = _U_INTERIOR * area * dt  # positive => heat flows a -> b

            flows.append(
                HeatFlow(
                    from_room=room_id,
                    to_room=neighbour_id,
                    flow_watts=round(q, 1),
                    shared_wall_length_m=round(wall_len, 2),
                )
            )

    return flows


def _shared_wall_length(poly_a: list[list[float]], poly_b: list[list[float]]) -> float:
    """Compute the length of the shared edge between two polygons.

    For axis-aligned rectangles this finds the overlapping segment on the
    shared edge defined by the two common vertices.
    """
    verts_a = {(_snap(p[0]), _snap(p[1])) for p in poly_a}
    verts_b = {(_snap(p[0]), _snap(p[1])) for p in poly_b}
    shared = sorted(verts_a & verts_b)

    if len(shared) < 2:
        return 0.0

    # Use first and last shared vertices to compute wall length
    p1, p2 = shared[0], shared[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return (dx * dx + dy * dy) ** 0.5


# ---------------------------------------------------------------------------
# 3. Net heat flow per room
# ---------------------------------------------------------------------------


def net_heat_flow_by_room(flows: list[HeatFlow]) -> dict[str, float]:
    """Aggregate heat flows to a net gain/loss per room.

    Positive means the room is gaining heat from neighbours.
    """
    net: dict[str, float] = {}
    for f in flows:
        net[f.from_room] = net.get(f.from_room, 0.0) - f.flow_watts
        net[f.to_room] = net.get(f.to_room, 0.0) + f.flow_watts
    return net


# ---------------------------------------------------------------------------
# 4. Root cause analysis
# ---------------------------------------------------------------------------


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
        pattern_ids = {p.pattern_id for p in patterns}

        # Open window: temp below average while power is high
        if metrics.temperature < avg_temp - 2.0 and metrics.power > 150.0:
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
                        f"average ({avg_temp:.1f}C) while power draw is "
                        f"{metrics.power:.0f}W — likely an open window"
                    ),
                    related_rooms=[],
                ),
            )

        # Solar gain: temp above average with low power
        if metrics.temperature > avg_temp + 2.0 and metrics.power < 80.0:
            causes.append(
                RootCause(
                    room_id=room_id,
                    pattern="solar_gain",
                    confidence=0.6,
                    explanation=(
                        f"Temperature {metrics.temperature:.1f}C is above "
                        f"average ({avg_temp:.1f}C) despite low power "
                        f"({metrics.power:.0f}W) — possible solar gain"
                    ),
                    related_rooms=[],
                ),
            )

        # Building-level insulation issue: same waste in adjacent rooms
        if "over_heating" in pattern_ids:
            affected_neighbours = [
                n for n in neighbours if any(p.pattern_id == "over_heating" for p in waste_patterns.get(n, []))
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
