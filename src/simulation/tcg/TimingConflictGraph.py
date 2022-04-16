from typing import Iterable, Set, Dict, Tuple, Optional

from simulation.intersection import Intersection
from simulation.vehicle import Vehicle

from .Vertex import Vertex, VertexState
from .Edge import Edge, EdgeType

class TimingConflictGraph:
    '''
    reference: Graph-based modeling, scheduling, and verification
               for intersection management of intelligent vehicles
    '''
    def __init__(
        self,
        vehicles: Set[Vehicle],
        intersection: Intersection
    ):
        self._vehicles: Set = vehicles
        self._intersection: Intersection = intersection
        self._V: Dict[Tuple[str, str], Vertex] = {}   # (vehicle id, cz id) -> Vertex
        self._E: Dict[Tuple[int, int], Edge] = {}     # (src vertex id, dst vertex id) -> Edge

        self.build_graph()

    @property
    def V(self) -> Iterable[Vertex]:
        return self._V.values()

    @property
    def E(self) -> Iterable[Edge]:
        return self._E.values()

    def build_graph(self) -> None:
        self._V: Dict[Tuple[str, str], Vertex] = {}
        self._E: Dict[Tuple[int, int], Edge] = {}

        for vehicle in self._vehicles:
            for cz_id in vehicle.trajectory:
                self._add_vertex(vehicle, cz_id)
            self._add_vertex(vehicle, f"${vehicle.id}")

        # Add type-1 edges
        for vehicle in self._vehicles:
            for idx, cz_id in enumerate(vehicle.trajectory[:-1]):
                self._add_edge_by_idx(
                    vehicle, cz_id,
                    vehicle, vehicle.trajectory[idx+1],
                    EdgeType.TYPE_1
                )
            self._add_edge_by_idx(
                vehicle, vehicle.trajectory[-1],
                vehicle, f"${vehicle.id}",
                EdgeType.TYPE_1,
                waiting_time=0
            )

        # Add type-2 edges
        for src_lane_id in self._intersection.src_lanes:
            vehicles_from_this_src_lane = [veh for veh in self._vehicles
                                           if veh.src_lane_id == src_lane_id]
            vehicles_from_this_src_lane.sort(key=lambda veh: veh.earliest_arrival_time)
            for idx, vehicle in enumerate(vehicles_from_this_src_lane[:-1]):
                for cz_id in vehicle.trajectory:
                    for later_vehicle in vehicles_from_this_src_lane[idx + 1:]:
                        if cz_id in later_vehicle.trajectory:
                            self._add_edge_by_idx(
                                vehicle, cz_id, later_vehicle, cz_id, EdgeType.TYPE_2)

        # Add type-3 edges
        self.add_undecided_type3_edges()

    @staticmethod
    def type3_edge_condition(v1: Vertex, v2: Vertex) -> bool:
        return v1.vehicle.id != v2.vehicle.id and v1.cz_id == v2.cz_id \
               and v1.vehicle.src_lane_id != v2.vehicle.src_lane_id

    def add_undecided_type3_edges(self) -> None:
        for v1 in self._V.values():
            for v2 in self._V.values():
                if self.type3_edge_condition(v1, v2):
                    self._add_edge_by_vtx(v1, v2, EdgeType.TYPE_3, decided=False)
                    self._add_edge_by_vtx(v2, v1, EdgeType.TYPE_3, decided=False)

    def print(self) -> None:
        for veh in sorted(self._vehicles, key=lambda veh: veh.id):
            print(f"* Vehicle {veh.id}, arrival time = {veh.earliest_arrival_time}")
            for cz_id in veh.trajectory:
                v = self.get_vertex_by_vehicle_cz_pair(veh, cz_id)
                print(f"    - ({v.vehicle.id}, {v.cz_id}): {v.state}; ", end="")
                print(f"p = {v.passing_time}, s = {v.entering_time}, s' = {v.earliest_entering_time} ", end="")
                print("parents: { ", end="")
                for in_e in v.in_edges:
                    print(f"({in_e.v_from.vehicle.id}, {in_e.v_from.cz_id}) ", end="")
                print("}")

    def start_execute(self, v: Vertex):
        key = (v.vehicle.id, v.cz_id)
        if key not in self._V or id(self._V[key]) != id(v):
            raise Exception("supplied vertex does not belongs to this graph")

        next_v = None
        w_e_to_next_v = 0
        for out_edge in v.out_edges:
            if out_edge.type == EdgeType.TYPE_1:
                next_v = out_edge.v_to
                w_e_to_next_v = out_edge.waiting_time

        for out_edge in v.out_edges:
            if out_edge.type == EdgeType.TYPE_3 and not out_edge.decided:
                disjunctive_edge = self._E[(out_edge.v_to.id, out_edge.v_from.id)]
                self._remove_edge(disjunctive_edge)
                out_edge.decided = True

                if next_v is not None:
                    self._add_edge_by_vtx(
                        next_v, out_edge.v_to,
                        waiting_time=out_edge.waiting_time - w_e_to_next_v - out_edge.waiting_time,
                        edge_type=EdgeType.TYPE_4
                    )

        v.state = VertexState.EXECUTING

    def finish_execute(self, v: Vertex):
        key = (v.vehicle.id, v.cz_id)
        if key not in self._V or id(self._V[key]) != id(v):
            raise Exception("supplied vertex does not belongs to this graph")
        v.state = VertexState.EXECUTED

    def reset_vertices_state(self) -> None:
        for v in self._V.values():
            v.state = VertexState.NON_EXECUTED
            v.earliest_entering_time = None
        self.add_undecided_type3_edges()

    def verify(self) -> None:
        for edge in self._E.values():
            if not edge.decided:
                raise Exception("undecided edge")
            s_from = edge.v_from.entering_time
            s_to = edge.v_to.entering_time
            if s_to < s_from + edge.v_from.passing_time + edge.waiting_time:
                raise Exception("timing constraint violated")

    def remove_vertex(self, v: Vertex) -> None:
        for in_e in v.in_edges:
            in_e.v_from.out_edges.remove(in_e)
            del self._E[(in_e.v_from.id, in_e.v_to.id)]

        for out_e in v.out_edges:
            out_e.v_to.in_edges.remove(out_e)
            del self._E[(out_e.v_from.id, out_e.v_to.id)]

        del self._V[(v.vehicle.id, v.cz_id)]

    def get_vertex_by_vehicle_cz_pair(self, vehicle: Vehicle, cz_id: str) -> Vertex:
        return self._V[(vehicle.id, cz_id)]

    def get_edge_by_vertex_pair(self, v_from: Vertex, v_to: Vertex) -> Edge:
        return self._E[(v_from.id, v_to.id)]

    def _add_vertex(self, vehicle: Vehicle, cz_id: str) -> None:
        if (vehicle.id, cz_id) not in self._V:
            v = Vertex(len(self._V), vehicle, cz_id)
            self._V[(vehicle.id, cz_id)] = v

    def _add_edge_by_idx(
        self,
        veh_from: Vehicle,
        cz_id_from: str,
        veh_to: Vehicle,
        cz_id_to: str,
        edge_type: EdgeType,
        waiting_time: Optional[int] = None,
        decided: bool = True
    ) -> None:
        v_from = self._V[(veh_from.id, cz_id_from)]
        v_to = self._V[(veh_to.id, cz_id_to)]
        self._add_edge_by_vtx(v_from, v_to, edge_type,
                              waiting_time=waiting_time, decided=decided)

    def _add_edge_by_vtx(
        self,
        v_from: Vertex,
        v_to: Vertex,
        edge_type: EdgeType,
        waiting_time: Optional[int] = None,
        decided: bool = True
    ) -> None:
        if (v_from.id, v_to.id) in self._E:
            return

        edge = Edge(len(self._E), v_from, v_to, edge_type,
                    waiting_time=waiting_time, decided=decided)
        self._E[(v_from.id, v_to.id)] = edge
        v_from.add_out_edge(edge)
        v_to.add_in_edge(edge)

    def _remove_edge(self, edge: Edge) -> None:
        key = (edge.v_from.id, edge.v_to.id)
        edge.v_from.remove_out_edge(edge)
        edge.v_to.remove_in_edge(edge)
        del self._E[key]
