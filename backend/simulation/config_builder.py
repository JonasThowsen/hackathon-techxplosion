"""Build room physics configs from building layout."""

import math

from core.models import BuildingLayout, Room
from simulation.room_config import (
    HVACConfig,
    RoomPhysicsConfig,
    WallConfig,
    estimate_thermal_mass,
)

# Default wall height
_WALL_HEIGHT_M = 3.0

# U-values W/(m²·K)
_U_INTERIOR = 1.5  # Interior walls
_U_EXTERIOR = 0.5  # Well-insulated exterior walls


def build_room_configs(
    layout: BuildingLayout,
    exterior_u_value: float = _U_EXTERIOR,
    interior_u_value: float = _U_INTERIOR,
    wall_height: float = _WALL_HEIGHT_M,
) -> dict[str, RoomPhysicsConfig]:
    """Build physics configs for all rooms in a building.

    Automatically detects:
    - Room volumes from polygon area
    - Shared walls between rooms (interior walls)
    - Exterior walls (edges not shared with other rooms)
    """
    configs: dict[str, RoomPhysicsConfig] = {}

    # Collect all rooms with their floor info
    all_rooms: list[tuple[Room, int]] = []
    for floor in layout.floors:
        for room in floor.rooms:
            all_rooms.append((room, floor.floor_index))

    # Build adjacency info
    adjacency = _compute_adjacency(all_rooms)

    for room, _ in all_rooms:
        area = _polygon_area(room.polygon)
        volume = area * wall_height
        thermal_mass = estimate_thermal_mass(volume)

        walls: list[WallConfig] = []

        # Process each edge of the polygon
        edges = _get_edges(room.polygon)
        for edge_start, edge_end in edges:
            edge_len = _distance(edge_start, edge_end)
            if edge_len < 0.01:
                continue

            # Find if this edge is shared with another room
            neighbor_id = _find_shared_edge_neighbor(room.id, edge_start, edge_end, adjacency)

            walls.append(
                WallConfig(
                    neighbor_id=neighbor_id or "exterior",
                    length_m=edge_len,
                    height_m=wall_height,
                    u_value=interior_u_value if neighbor_id else exterior_u_value,
                )
            )

        configs[room.id] = RoomPhysicsConfig(
            room_id=room.id,
            volume_m3=volume,
            thermal_mass_j_per_k=thermal_mass,
            walls=walls,
            hvac=HVACConfig(),  # Default HVAC settings
        )

    return configs


def _polygon_area(polygon: list[list[float]]) -> float:
    """Calculate area of a polygon using the shoelace formula."""
    n = len(polygon)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def _get_edges(polygon: list[list[float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Get all edges of a polygon as (start, end) tuples."""
    edges: list[tuple[tuple[float, float], tuple[float, float]]] = []
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        start = (polygon[i][0], polygon[i][1])
        end = (polygon[j][0], polygon[j][1])
        edges.append((start, end))
    return edges


def _distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def _snap(v: float, epsilon: float = 0.01) -> float:
    """Snap a coordinate to grid."""
    return round(v / epsilon) * epsilon


def _compute_adjacency(rooms: list[tuple[Room, int]]) -> dict[str, list[tuple[str, set[tuple[float, float]]]]]:
    """Compute which rooms share edges.

    Returns: room_id -> [(neighbor_id, shared_vertices), ...]
    """
    # Collect vertices per room (snapped to grid)
    room_verts: dict[str, set[tuple[float, float]]] = {}
    for room, _ in rooms:
        verts = {(_snap(p[0]), _snap(p[1])) for p in room.polygon}
        room_verts[room.id] = verts

    # Find shared vertices between rooms
    adjacency: dict[str, list[tuple[str, set[tuple[float, float]]]]] = {}
    room_ids = list(room_verts.keys())

    for i, room_a in enumerate(room_ids):
        adjacency[room_a] = []
        for room_b in room_ids[i + 1 :]:
            shared = room_verts[room_a] & room_verts[room_b]
            if len(shared) >= 2:
                adjacency[room_a].append((room_b, shared))
                adjacency.setdefault(room_b, []).append((room_a, shared))

    return adjacency


def _find_shared_edge_neighbor(
    room_id: str,
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
    adjacency: dict[str, list[tuple[str, set[tuple[float, float]]]]],
) -> str | None:
    """Find if an edge is shared with a neighbor room."""
    snapped_start = (_snap(edge_start[0]), _snap(edge_start[1]))
    snapped_end = (_snap(edge_end[0]), _snap(edge_end[1]))
    edge_verts = {snapped_start, snapped_end}

    neighbors = adjacency.get(room_id, [])
    for neighbor_id, shared_verts in neighbors:
        # Check if both edge vertices are in the shared set
        if edge_verts <= shared_verts:
            return neighbor_id

    return None
