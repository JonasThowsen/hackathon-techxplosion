"""Room adjacency detection based on polygon geometry."""

from core.models import Floor, Room

# Two vertices are "the same" if within this distance (metres).
_EPSILON = 0.01


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


def find_vertical_neighbours(floors: list[Floor]) -> dict[str, list[str]]:
    """Find rooms on adjacent floors that overlap (share floor/ceiling).

    Two rooms are vertical neighbours if they are on consecutive floors
    and their polygons share at least two vertices (overlapping footprint).
    """
    neighbours: dict[str, list[str]] = {}
    sorted_floors = sorted(floors, key=lambda f: f.floor_index)

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


def _snap(v: float) -> float:
    """Snap a coordinate to the nearest epsilon grid."""
    return round(v / _EPSILON) * _EPSILON
