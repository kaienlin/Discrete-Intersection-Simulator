import enum
from typing import Set, Union, Optional

from simulation.tcg.Edge import Edge, EdgeType
from simulation.vehicle import Vehicle

class VertexState(enum.Enum):
    NON_EXECUTED = enum.auto()
    EXECUTING = enum.auto()
    EXECUTED = enum.auto()

    def __str__(self) -> str:
        return self.name


class Vertex:
    def __init__(
        self,
        _id: int,
        vehicle: Vehicle,
        cz_id: str,
        passing_time: Optional[int] = None
    ) -> None:
        self.id: int = _id
        self.vehicle: Vehicle = vehicle
        self.cz_id: str = cz_id

        self.out_edges: Set[Edge] = set()
        self.in_edges: Set[Edge] = set()
        self.entering_time: int = 0
        if passing_time is None:
            self.passing_time: int = vehicle.vertex_passing_time
        else:
            self.passing_time: int = passing_time

        self.earliest_entering_time: Union[None, int] = None
        self.entering_time_wo_delay: int = 0
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

    def remove_out_edge(self, edge: Edge) -> None:
        self.out_edges.remove(edge)

    def remove_in_edge(self, edge: Edge) -> None:
        self.in_edges.remove(edge)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return self.id == other.id
