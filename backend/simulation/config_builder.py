"""Build room physics configs from building layout."""

import logging
import math

from core.models import BuildingLayout, Room
from simulation.room_config import (
    HVACConfig,
    RoomPhysicsConfig,
    WallConfig,
    WindowConfig,
    estimate_thermal_mass,
)

logger = logging.getLogger(__name__)

# Default wall height
_WALL_HEIGHT_M = 3.0

# U-values W/(m²·K)
_U_INTERIOR = 1.5  # Interior walls
_U_EXTERIOR = 0.5  # Well-insulated exterior walls
_U_FLOOR_CEILING = 0.4  # Concrete slab between floors

# Window sizing: 30% of exterior wall area
_WINDOW_FRACTION = 0.3


def _wall_orientation(edge_start: tuple[float, float], edge_end: tuple[float, float]) -> str | None:
    """Determine cardinal orientation of an exterior wall edge.

    Returns N/S/E/W or None if the wall is not axis-aligned.
    Convention: the orientation is the direction the wall *faces* (outward normal).
    """
    dx = edge_end[0] - edge_start[0]
    dy = edge_end[1] - edge_start[1]

    # Horizontal wall (dy ≈ 0)
    if abs(dy) < 0.01 and abs(dx) > 0.01:
        # Polygon is wound clockwise in screen coords (y increases downward).
        # A top edge (going left-to-right, dx > 0) has outward normal pointing up → North.
        # A bottom edge (going right-to-left, dx < 0) has outward normal pointing down → South.
        return "N" if dx > 0 else "S"

    # Vertical wall (dx ≈ 0)
    if abs(dx) < 0.01 and abs(dy) > 0.01:
        # A right edge (going top-to-bottom, dy > 0) has outward normal pointing right → East.
        # A left edge (going bottom-to-top, dy < 0) has outward normal pointing left → West.
        return "E" if dy > 0 else "W"

    return None


def build_room_configs(
    layout: BuildingLayout,
    exterior_u_value: float = _U_EXTERIOR,
    interior_u_value: float = _U_INTERIOR,
    floor_ceiling_u_value: float = _U_FLOOR_CEILING,
    wall_height: float = _WALL_HEIGHT_M,
) -> dict[str, RoomPhysicsConfig]:
    """Build physics configs for all rooms in a building.

    Automatically detects:
    - Room volumes from polygon area
    - Shared walls between rooms (interior walls)
    - Exterior walls (edges not shared with other rooms)
    - Windows on exterior walls with correct orientation
    - Floor/ceiling heat transfer to rooms directly above/below
    """
    configs: dict[str, RoomPhysicsConfig] = {}

    # Collect all rooms with their floor info
    all_rooms: list[tuple[Room, int]] = []
    for floor in layout.floors:
        for room in floor.rooms:
            all_rooms.append((room, floor.floor_index))

    # Build adjacency info (same-floor walls)
    adjacency = _compute_adjacency(all_rooms)

    # Build vertical adjacency (floor/ceiling between adjacent floors)
    vertical = _compute_vertical_adjacency(layout)

    for room, _ in all_rooms:
        area = _polygon_area(room.polygon)
        volume = area * wall_height
        thermal_mass = estimate_thermal_mass(volume)

        walls: list[WallConfig] = []
        windows: list[WindowConfig] = []

        # Process each edge of the polygon (horizontal walls)
        edges = _get_edges(room.polygon)
        for edge_start, edge_end in edges:
            edge_len = _distance(edge_start, edge_end)
            if edge_len < 0.01:
                continue

            # Find if this edge is shared with another room
            neighbor_id = _find_shared_edge_neighbor(room.id, edge_start, edge_end, adjacency)

            is_exterior = neighbor_id is None
            walls.append(
                WallConfig(
                    neighbor_id=neighbor_id or "exterior",
                    length_m=edge_len,
                    height_m=wall_height,
                    u_value=interior_u_value if neighbor_id else exterior_u_value,
                )
            )

            # Add window for exterior walls
            if is_exterior:
                orientation = _wall_orientation(edge_start, edge_end)
                if orientation is not None:
                    window_area = edge_len * wall_height * _WINDOW_FRACTION
                    windows.append(
                        WindowConfig(
                            orientation=orientation,
                            area_m2=window_area,
                        )
                    )

        # Add floor/ceiling "walls" for vertical heat transfer
        for v_neighbour_id in vertical.get(room.id, []):
            # Use sqrt(area) as effective length so that area_m2 = polygon area
            effective_length = math.sqrt(area)
            walls.append(
                WallConfig(
                    neighbor_id=v_neighbour_id,
                    length_m=effective_length,
                    height_m=effective_length,
                    u_value=floor_ceiling_u_value,
                )
            )

        # Size HVAC to room heat loss: rooms with more exterior exposure
        # need higher heating gain to maintain target temperature.
        # With proportional control (power = gain * error), the equilibrium
        # satisfies: gain * (target - T) = ext_conductance * (T - T_outside).
        # To keep undershoot below ~0.5 C at design dT ~ 18 K:
        #   gain >= ext_conductance * 18 / 0.5 = ext_conductance * 36
        exterior_conductance = sum(w.u_value * w.area_m2 for w in walls if w.neighbor_id == "exterior")
        min_gain = exterior_conductance * 36
        hvac_gain = max(HVACConfig.heating_gain, min_gain)

        configs[room.id] = RoomPhysicsConfig(
            room_id=room.id,
            volume_m3=volume,
            thermal_mass_j_per_k=thermal_mass,
            walls=walls,
            windows=windows,
            hvac=HVACConfig(heating_gain=hvac_gain),
        )

    # Log thermal properties for every room
    for room_id, rc in configs.items():
        logger.info(
            "Room %s: thermal_mass=%.1f kJ/K",
            room_id,
            rc.thermal_mass_j_per_k / 1000,
        )
        for wall in rc.walls:
            conductance = wall.u_value * wall.area_m2  # W/K
            logger.info(
                "  -> %s: U=%.2f W/m²K, A=%.1f m², conductance=%.1f W/K",
                wall.neighbor_id,
                wall.u_value,
                wall.area_m2,
                conductance,
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
    # Collect vertices per room (snapped to grid) and track floor membership
    room_verts: dict[str, set[tuple[float, float]]] = {}
    room_floor: dict[str, int] = {}
    for room, floor_index in rooms:
        verts = {(_snap(p[0]), _snap(p[1])) for p in room.polygon}
        room_verts[room.id] = verts
        room_floor[room.id] = floor_index

    # Find shared vertices between rooms (same floor only)
    adjacency: dict[str, list[tuple[str, set[tuple[float, float]]]]] = {}
    room_ids = list(room_verts.keys())

    for i, room_a in enumerate(room_ids):
        adjacency.setdefault(room_a, [])
        for room_b in room_ids[i + 1 :]:
            if room_floor[room_a] != room_floor[room_b]:
                continue
            shared = room_verts[room_a] & room_verts[room_b]
            if len(shared) >= 2:
                adjacency[room_a].append((room_b, shared))
                adjacency.setdefault(room_b, []).append((room_a, shared))

    return adjacency


def _compute_vertical_adjacency(layout: BuildingLayout) -> dict[str, list[str]]:
    """Find rooms on adjacent floors with overlapping footprints (floor/ceiling)."""
    neighbours: dict[str, list[str]] = {}
    sorted_floors = sorted(layout.floors, key=lambda f: f.floor_index)

    for idx in range(len(sorted_floors) - 1):
        lower = sorted_floors[idx]
        upper = sorted_floors[idx + 1]
        for room_a in lower.rooms:
            verts_a = {(_snap(p[0]), _snap(p[1])) for p in room_a.polygon}
            for room_b in upper.rooms:
                verts_b = {(_snap(p[0]), _snap(p[1])) for p in room_b.polygon}
                shared = verts_a & verts_b
                if len(shared) >= 2:
                    neighbours.setdefault(room_a.id, []).append(room_b.id)
                    neighbours.setdefault(room_b.id, []).append(room_a.id)

    return neighbours


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
