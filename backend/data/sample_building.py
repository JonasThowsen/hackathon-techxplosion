"""Sample building layout for testing."""

from core.models import BuildingLayout, Floor, Room


def create_sample_building() -> BuildingLayout:
    """Create a hardcoded sample student housing building layout."""
    return BuildingLayout(
        id="building-1",
        name="Student Housing A",
        width_m=30.0,
        height_m=18.0,
        floors=[
            _ground_floor(),
            _first_floor(),
            _second_floor(),
        ],
    )


def _ground_floor() -> Floor:
    """Ground floor: lobby, kitchen, common room, laundry, study room, storage."""
    return Floor(
        floor_index=0,
        label="Ground Floor",
        rooms=[
            Room(id="r-001", name="Lobby", polygon=[[0, 0], [10, 0], [10, 9], [0, 9]]),
            Room(id="r-002", name="Kitchen", polygon=[[10, 0], [20, 0], [20, 9], [10, 9]]),
            Room(id="r-003", name="Common Room", polygon=[[20, 0], [30, 0], [30, 9], [20, 9]]),
            Room(id="r-004", name="Laundry", polygon=[[0, 9], [10, 9], [10, 18], [0, 18]]),
            Room(id="r-005", name="Study Room A", polygon=[[10, 9], [20, 9], [20, 18], [10, 18]]),
            Room(id="r-006", name="Storage", polygon=[[20, 9], [30, 9], [30, 18], [20, 18]]),
        ],
    )


def _first_floor() -> Floor:
    """First floor: bedrooms and a shared bathroom."""
    return Floor(
        floor_index=1,
        label="First Floor",
        rooms=[
            Room(id="r-101", name="Bedroom 101", polygon=[[0, 0], [10, 0], [10, 9], [0, 9]]),
            Room(id="r-102", name="Bedroom 102", polygon=[[10, 0], [20, 0], [20, 9], [10, 9]]),
            Room(id="r-103", name="Bedroom 103", polygon=[[20, 0], [30, 0], [30, 9], [20, 9]]),
            Room(id="r-104", name="Bedroom 104", polygon=[[0, 9], [10, 9], [10, 18], [0, 18]]),
            Room(id="r-105", name="Shared Bathroom", polygon=[[10, 9], [20, 9], [20, 18], [10, 18]]),
            Room(id="r-106", name="Bedroom 105", polygon=[[20, 9], [30, 9], [30, 18], [20, 18]]),
        ],
    )


def _second_floor() -> Floor:
    """Second floor: bedrooms and study room."""
    return Floor(
        floor_index=2,
        label="Second Floor",
        rooms=[
            Room(id="r-201", name="Bedroom 201", polygon=[[0, 0], [10, 0], [10, 9], [0, 9]]),
            Room(id="r-202", name="Bedroom 202", polygon=[[10, 0], [20, 0], [20, 9], [10, 9]]),
            Room(id="r-203", name="Bedroom 203", polygon=[[20, 0], [30, 0], [30, 9], [20, 9]]),
            Room(id="r-204", name="Bedroom 204", polygon=[[0, 9], [15, 9], [15, 18], [0, 18]]),
            Room(id="r-205", name="Study Room B", polygon=[[15, 9], [30, 9], [30, 18], [15, 18]]),
        ],
    )


SAMPLE_BUILDING = create_sample_building()
