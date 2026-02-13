# CLAUDE.md

## Project

Hackathon project: IoT-based energy waste detection and automatic control for buildings. This is the backend -- a FastAPI service exposing a REST API for energy metrics, waste detection, and control actions.

## Stack

- Python 3.14, managed with uv
- FastAPI for the REST API
- Linting: ruff
- Type checking: basedpyright (strict mode)
- Task runner: just

## Commands

- `just check` -- run lint + typecheck
- `just lint` -- ruff check --fix + format
- `just typecheck` -- basedpyright
- `uv run main.py` -- run the project

## Architecture: Polymorphic Hierarchy

The system uses an abstract base class `EnergyZone` that every level of the building hierarchy implements. This is the central design pattern -- all code should follow it.

### The ABC

```python
from abc import ABC, abstractmethod

class EnergyZone(ABC):
    """Every level of the hierarchy implements this interface."""

    @abstractmethod
    def collect_metrics(self) -> Metrics: ...

    @abstractmethod
    def identify_waste(self) -> list[WastePattern]: ...

    @abstractmethod
    def act(self) -> list[Action]: ...

    @abstractmethod
    def visualize(self) -> Visualization: ...
```

### Sensors and Coordinates

Sensors are the raw data source. Each sensor has a physical position and a floor number:

```python
@dataclass
class Sensor:
    id: str
    kind: SensorKind  # temperature, occupancy, power, light
    x: float
    y: float
    floor: int
```

The hierarchy is derived from sensor positions, not hardcoded:
- A **Room** is a group of sensors within a spatial boundary on the same floor.
- A **Floor** is all sensors sharing a floor number.
- A **Building** is all sensors.

### Hierarchy

- `Room(EnergyZone)` -- the atomic unit. Groups sensors by spatial proximity. Actions are concrete (turn off lights, lower thermostat). This is where waste is detected and fixed.
- `Floor(EnergyZone)` -- holds a `list[Room]`. Aggregates room metrics upward. Handles resident-facing visualization and nudges.
- `Building(EnergyZone)` -- holds a `list[Floor]`. Aggregates floor metrics upward. Handles operator dashboard and building-wide patterns.

Composition, not deep inheritance: Floor composes Rooms, Building composes Floors. The aggregation logic lives in each level's implementation.

### Key rules

- Any function that operates on an `EnergyZone` must work at every level. Never type-check for Room/Floor/Building -- use the abstract interface.
- Data flows upward: rooms produce metrics, floors aggregate them, buildings aggregate further.
- Actions flow downward: building-level decisions cascade to floors, floor-level decisions cascade to rooms.
- Sensor data is only read at the Room level. Floor and Building never touch sensors directly.

### Device Mesh

IoT devices in proximity communicate directly via a mesh network. Neighboring devices share state and react using a local rule engine (if neighbor reports X, do Y). Proximity is derived from sensor coordinates -- no manual neighbor configuration needed. The central system collects data for aggregation, but actions happen at the edge.

## API Design

This backend exposes a FastAPI service. A separate React frontend consumes it. The backend has no knowledge of the frontend.

- REST endpoints for initial data (building structure, sensor positions, configuration).
- A single WebSocket (`/ws`) streams the full building state to connected clients. The backend pushes the complete metrics tree (building > floors > rooms) on each tick. The frontend filters client-side based on what view the user is looking at.
- Use Pydantic models for all request/response schemas. These can mirror or wrap the core data types.
- Keep route handlers thin -- they call into the EnergyZone hierarchy, not the other way around.

## Code style

- Use `dataclass` or `NamedTuple` for internal data types (`Metrics`, `WastePattern`, `Action`, etc.). Use Pydantic `BaseModel` for API schemas.
- Keep type annotations on everything -- basedpyright strict will enforce this.
- No classes where a function will do. Only use classes for the EnergyZone hierarchy and data types.
- Prefer composition over inheritance beyond the single ABC layer.
