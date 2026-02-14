"""Thermal graph construction and heat flow mapping.

This module treats the building as a weighted graph where:
- Nodes are rooms
- Edges are thermal connections weighted by conductance

Heat flows along edges: Q_ij = G_ij * (T_i - T_j)
"""

import numpy as np
from numpy.typing import NDArray

from analysis.thermal.types import (
    EdgeSpec,
    HeatFlowEdge,
    HeatFlowSnapshot,
    ThermalBridge,
    ThermalGraph,
    ThermalParameters,
)


def build_graph_from_adjacency(
    room_ids: list[str],
    adjacency: dict[str, list[str]],
    wall_areas: dict[tuple[str, str], float] | None = None,
    exterior_rooms: set[str] | None = None,
    default_wall_area: float = 9.0,
) -> ThermalGraph:
    """Build a ThermalGraph from room adjacency information.

    Args:
        room_ids: List of all room IDs
        adjacency: Adjacency dict mapping room_id -> list of neighbor room_ids
        wall_areas: Optional wall areas (room_a, room_b) -> area in m²
        exterior_rooms: Set of room IDs that have exterior walls
        default_wall_area: Default wall area if not specified (m²)

    Returns:
        ThermalGraph ready for estimation
    """
    edges: list[EdgeSpec] = []
    seen_pairs: set[tuple[str, str]] = set()

    # Build interior edges
    for room_id, neighbors in adjacency.items():
        for neighbor_id in neighbors:
            if neighbor_id not in room_ids:
                continue

            # Create sorted pair for deduplication
            pair = _sorted_pair(room_id, neighbor_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            # Get wall area
            area = default_wall_area
            if wall_areas:
                area = wall_areas.get(pair, wall_areas.get((pair[1], pair[0]), default_wall_area))

            edges.append(
                EdgeSpec(
                    node_a=pair[0],
                    node_b=pair[1],
                    wall_area_m2=area,
                    is_exterior=False,
                )
            )

    # Build exterior edges
    if exterior_rooms:
        for room_id in exterior_rooms:
            if room_id in room_ids:
                edges.append(
                    EdgeSpec(
                        node_a=room_id,
                        node_b="exterior",
                        wall_area_m2=default_wall_area,
                        is_exterior=True,
                    )
                )

    return ThermalGraph(node_ids=room_ids, edges=edges)


def compute_heat_flow(
    from_temp: float,
    to_temp: float,
    conductance: float,
) -> float:
    """Compute heat flow between two nodes.

    Heat flows from higher to lower temperature.

    Args:
        from_temp: Temperature of source node (°C)
        to_temp: Temperature of destination node (°C)
        conductance: Thermal conductance G (W/K)

    Returns:
        Heat flow Q (W), positive means heat flows from -> to
    """
    return conductance * (from_temp - to_temp)


def compute_heat_flow_snapshot(
    graph: ThermalGraph,
    parameters: ThermalParameters,
    room_temps: dict[str, float],
    external_temp: float,
    timestep: int = 0,
) -> HeatFlowSnapshot:
    """Compute heat flow along all edges at a single timestep.

    Args:
        graph: The thermal graph
        parameters: Estimated thermal parameters
        room_temps: Current temperature of each room
        external_temp: Current external temperature
        timestep: Timestep index for the snapshot

    Returns:
        Heat flow snapshot with flows along each edge
    """
    flows: list[HeatFlowEdge] = []
    net_by_room: dict[str, float] = dict.fromkeys(graph.node_ids, 0.0)

    for edge in graph.edges:
        if edge.is_exterior:
            # Exterior edge: heat loss to outside
            room_id = edge.node_a if edge.node_a in graph.node_ids else edge.node_b
            room_params = parameters.rooms.get(room_id)
            if room_params is None or room_params.exterior_conductance is None:
                continue

            conductance = room_params.exterior_conductance
            room_temp = room_temps.get(room_id, 20.0)
            flow = compute_heat_flow(room_temp, external_temp, conductance)

            flows.append(
                HeatFlowEdge(
                    from_room=room_id,
                    to_room="exterior",
                    flow_watts=flow,
                    conductance=conductance,
                )
            )

            # Negative because heat is leaving the room
            net_by_room[room_id] -= flow
        else:
            # Interior edge
            conductance = parameters.get_conductance(edge.node_a, edge.node_b)
            temp_a = room_temps.get(edge.node_a, 20.0)
            temp_b = room_temps.get(edge.node_b, 20.0)
            flow = compute_heat_flow(temp_a, temp_b, conductance)

            flows.append(
                HeatFlowEdge(
                    from_room=edge.node_a,
                    to_room=edge.node_b,
                    flow_watts=flow,
                    conductance=conductance,
                )
            )

            # Update net heat flow for both rooms
            net_by_room[edge.node_a] -= flow  # Heat leaving room A
            net_by_room[edge.node_b] += flow  # Heat entering room B

    return HeatFlowSnapshot(
        timestep=timestep,
        flows=flows,
        net_by_room=net_by_room,
    )


def compute_heat_flow_trajectory(
    graph: ThermalGraph,
    parameters: ThermalParameters,
    room_temp_series: dict[str, NDArray[np.float64]],
    external_temps: NDArray[np.float64],
) -> list[HeatFlowSnapshot]:
    """Compute heat flow snapshots for all timesteps.

    Args:
        graph: The thermal graph
        parameters: Estimated thermal parameters
        room_temp_series: Temperature time series for each room
        external_temps: External temperature time series

    Returns:
        List of heat flow snapshots, one per timestep
    """
    n_timesteps = len(external_temps)
    snapshots: list[HeatFlowSnapshot] = []

    for t in range(n_timesteps):
        room_temps = {room_id: float(temps[t]) for room_id, temps in room_temp_series.items()}
        snapshot = compute_heat_flow_snapshot(
            graph=graph,
            parameters=parameters,
            room_temps=room_temps,
            external_temp=float(external_temps[t]),
            timestep=t,
        )
        snapshots.append(snapshot)

    return snapshots


def identify_thermal_bridges(
    parameters: ThermalParameters,
    graph: ThermalGraph,
    exterior_threshold_low: float = 3.0,
    exterior_threshold_high: float = 8.0,
    interior_threshold_low: float = 10.0,
    interior_threshold_high: float = 25.0,
) -> list[ThermalBridge]:
    """Identify thermal bridges (high conductance weak points).

    Thermal bridges are locations where heat loss is abnormally high,
    indicating poor insulation or structural weak points.

    Args:
        parameters: Estimated thermal parameters
        graph: The thermal graph
        exterior_threshold_low: Conductance above this is 'medium' severity for exterior
        exterior_threshold_high: Conductance above this is 'high' severity for exterior
        interior_threshold_low: Conductance above this is 'medium' severity for interior
        interior_threshold_high: Conductance above this is 'high' severity for interior

    Returns:
        List of identified thermal bridges, sorted by severity
    """
    _ = graph  # Reserved for topology-based analysis
    bridges: list[ThermalBridge] = []

    # Check exterior walls
    for room_id, room_params in parameters.rooms.items():
        if room_params.exterior_conductance is not None:
            g = room_params.exterior_conductance
            if g > exterior_threshold_low:
                severity = "high" if g > exterior_threshold_high else "medium"
                bridges.append(
                    ThermalBridge(
                        room_ids=(room_id, "exterior"),
                        is_exterior=True,
                        conductance=g,
                        severity=severity,
                    )
                )

    # Check interior walls
    for (room_a, room_b), g in parameters.conductances.items():
        if g > interior_threshold_low:
            severity = "high" if g > interior_threshold_high else "medium"
            bridges.append(
                ThermalBridge(
                    room_ids=(room_a, room_b),
                    is_exterior=False,
                    conductance=g,
                    severity=severity,
                )
            )

    # Sort by severity (high first) then by conductance
    severity_order = {"high": 0, "medium": 1, "low": 2}
    bridges.sort(key=lambda b: (severity_order[b.severity], -b.conductance))

    return bridges


def rank_rooms_by_heat_loss(
    snapshots: list[HeatFlowSnapshot],
) -> list[tuple[str, float]]:
    """Rank rooms by average net heat loss.

    Args:
        snapshots: Heat flow snapshots over time

    Returns:
        List of (room_id, avg_heat_loss) sorted by loss (highest first)
        Positive values indicate net heat loss.
    """
    if not snapshots:
        return []

    # Accumulate net heat flow per room
    totals: dict[str, float] = {}
    for snapshot in snapshots:
        for room_id, net_flow in snapshot.net_by_room.items():
            # net_flow is positive when room gains heat, negative when loses
            totals[room_id] = totals.get(room_id, 0.0) - net_flow

    # Average over timesteps
    n = len(snapshots)
    averages = [(room_id, total / n) for room_id, total in totals.items()]

    # Sort by heat loss (highest first)
    averages.sort(key=lambda x: -x[1])

    return averages


def build_conductance_matrix(
    graph: ThermalGraph,
    parameters: ThermalParameters,
) -> NDArray[np.float64]:
    """Build the conductance matrix G for the thermal network.

    The conductance matrix G is defined such that:
    - G[i,j] = conductance between room i and room j (off-diagonal)
    - G[i,i] = -sum of all conductances from room i (diagonal)

    This matrix appears in the state-space form: C * dT/dt = G @ T + P

    Args:
        graph: The thermal graph
        parameters: Thermal parameters with conductances

    Returns:
        Conductance matrix G, shape (n_rooms, n_rooms)
    """
    n = len(graph.node_ids)
    G = np.zeros((n, n), dtype=np.float64)

    for edge in graph.edges:
        if edge.is_exterior:
            # Exterior edge contributes to diagonal only
            room_id = edge.node_a if edge.node_a in graph.node_ids else edge.node_b
            room_params = parameters.rooms.get(room_id)
            if room_params and room_params.exterior_conductance:
                i = graph.node_index(room_id)
                G[i, i] -= room_params.exterior_conductance
        else:
            # Interior edge
            i = graph.node_index(edge.node_a)
            j = graph.node_index(edge.node_b)
            g = parameters.get_conductance(edge.node_a, edge.node_b)

            G[i, j] = g
            G[j, i] = g
            G[i, i] -= g
            G[j, j] -= g

    return G


def _sorted_pair(a: str, b: str) -> tuple[str, str]:
    """Return a sorted pair of strings."""
    if a <= b:
        return (a, b)
    return (b, a)
