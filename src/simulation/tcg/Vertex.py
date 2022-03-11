import enum
from typing import Set

from simulation.tcg.Edge import Edge, EdgeType
from simulation.vehicle import Vehicle

class VertexState(enum.Enum):
    NON_EXECUTED = enum.auto()
    EXECUTING = enum.auto()
    EXECUTED = enum.auto()

    def __str__(self) -> str:
        return self.name


class Vertex:
    def __init__(self, _id: int, vehicle: Vehicle, cz_id: str) -> None:
        self.id: int = _id
        self.vehicle: Vehicle = vehicle
        self.cz_id: str = cz_id

        self.out_edges: Set[Edge] = set()
        self.in_edges: Set[Edge] = set()
        self.entering_time: int = 0
        self.passing_time: int = vehicle.vertex_passing_time

        self.earliest_entering_time = 0
        self.state: VertexState = VertexState.NON_EXECUTED

    def get_consumed_time(self) -> int:
        for out_e in self.out_edges:
            if out_e.type == EdgeType.TYPE_1:
                return self.passing_time + out_e.waiting_time
        return self.passing_time

    def add_out_edge(self, edge: Edge) -> None:
        self.out_edges.add(edge)

    def add_in_edge(self, edge: Edge) -> None:
        self.in_edges.add(edge)
    
    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return self.id == other.id

