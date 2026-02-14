"""Heat flow calculations between adjacent rooms."""

from dataclasses import dataclass

from core.models import Room
from core.zones.base import Metrics

# Two vertices are "the same" if within this distance (metres).
_EPSILON: float = 0.01

# Interior wall U-value in W/(m^2*K)
_U_INTERIOR: float = 1.5

# Assumed floor-to-ceiling height in metres
_WALL_HEIGHT_M: float = 3.0


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
    """Compute the length of the shared edge between two polygons."""
    verts_a = {(_snap(p[0]), _snap(p[1])) for p in poly_a}
    verts_b = {(_snap(p[0]), _snap(p[1])) for p in poly_b}
    shared = sorted(verts_a & verts_b)

    if len(shared) < 2:
        return 0.0

    p1, p2 = shared[0], shared[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return (dx * dx + dy * dy) ** 0.5


def _snap(v: float) -> float:
    """Snap a coordinate to the nearest epsilon grid."""
    return round(v / _EPSILON) * _EPSILON


def net_heat_flow_by_room(flows: list[HeatFlow]) -> dict[str, float]:
    """Aggregate heat flows to a net gain/loss per room.

    Positive means the room is gaining heat from neighbours.
    """
    net: dict[str, float] = {}
    for f in flows:
        net[f.from_room] = net.get(f.from_room, 0.0) - f.flow_watts
        net[f.to_room] = net.get(f.to_room, 0.0) + f.flow_watts
    return net
