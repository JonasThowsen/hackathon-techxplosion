"""Room adjacency detection based on polygon geometry."""

from core.models import Room

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


def _snap(v: float) -> float:
    """Snap a coordinate to the nearest epsilon grid."""
    return round(v / _EPSILON) * _EPSILON
