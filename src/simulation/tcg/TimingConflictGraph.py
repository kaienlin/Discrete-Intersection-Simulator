from typing import Iterable, Set, Dict, Tuple

from simulation.intersection import Intersection
from simulation.vehicle import Vehicle

from .Vertex import Vertex, VertexState
from .Edge import Edge, EdgeType

class TimingConflictGraph:
    def __init__(
        self,
        vehicles: Set,
        intersection: Intersection
    ):
        self.__vehicles: Set = vehicles
        self.__intersection: Intersection = intersection
        self.build_graph()

    def build_graph(self) -> None:
        self.__V: Dict[Tuple[str, str], Vertex] = dict()
        self.__E: Dict[Tuple[int, int], Edge] = dict()

        for vehicle in self.__vehicles:
            for cz_id in vehicle.trajectory:
                self.__add_vertex(vehicle, cz_id)
        
        # Add type-1 edges
        for vehicle in self.__vehicles:
            for idx, cz_id in enumerate(vehicle.trajectory[:-1]):
                self.__add_edge_by_idx(
                    vehicle, cz_id,
                    vehicle, vehicle.trajectory[idx+1],
                    EdgeType.TYPE_1
                )
    
        # Add type-2 edges
        for src_lane_id in self.__intersection.src_lanes:
            vehicles = [veh for veh in self.__vehicles if veh.src_lane_id == src_lane_id]
            vehicles.sort(key=lambda veh: veh.earliest_arrival_time)
            for idx, veh in enumerate(vehicles[:-1]):
                for cz_id in veh.trajectory:
                    for later_veh in vehicles[idx + 1:]:
                        if cz_id in later_veh.trajectory:
                            self.__add_edge_by_idx(veh, cz_id, later_veh, cz_id, EdgeType.TYPE_2)

    def print(self) -> None:
        for veh in self.__vehicles:
            print(f"* Vehicle {veh.id}")
            for cz_id in veh.trajectory:
                v = self.get_v_by_idx(veh, cz_id)
                print(f"    - ({v.vehicle.id}, {v.cz_id}): {v.state}; s = {v.entering_time} ", end = "")
                print("parents: { ", end="")
                for in_e in v.in_edges:
                    print(f"({in_e.v_from.vehicle.id}, {in_e.v_from.cz_id}) ", end="")
                print("}")

    def add_vehicle(self, vehicle: Vehicle) -> None:
        self.__vehicles.add(vehicle)

        for cz_id in vehicle.trajectory:
            self.__add_vertex(vehicle, cz_id)

        # Add type-1 edges
        for idx, cz_id in enumerate(vehicle.trajectory[:-1]):
            self.__add_edge_by_idx(
                vehicle, cz_id,
                vehicle, vehicle.trajectory[idx+1],
                EdgeType.TYPE_1
            )

        # Add type-2 edges
        same_lane_vehicles = [veh for veh in self.__vehicles if veh.src_lane_id == vehicle.src_lane_id]
        for veh2 in same_lane_vehicles:
            if veh2.id == vehicle.id:
                continue
            for cz_id in vehicle.trajectory:
                if cz_id in veh2.trajectory:
                    self.__add_edge_by_idx(vehicle, cz_id, veh2, cz_id, EdgeType.TYPE_2)

        # Add type-3 edges
        for cz_id in vehicle.trajectory:
            v = self.__V[(vehicle.id, cz_id)]
            for u in self.__V.values():
                if u.cz_id == cz_id and (u.state == VertexState.EXECUTING or u.state == VertexState.EXECUTED):
                    self.__add_edge_by_vtx(u, v, EdgeType.TYPE_3)
        

    def start_execute(self, v: Vertex):
        key = (v.vehicle.id, v.cz_id)
        if key not in self.__V or id(self.__V[key]) != id(v):
            raise Exception("[TimingConflictGraph.start_execute] supplied vertex does not belongs to this graph")
        v.state = VertexState.EXECUTING
        for u in self.__V.values():
            if u.cz_id == v.cz_id \
               and u.state == VertexState.NON_EXECUTED \
               and u.vehicle.src_lane_id != v.vehicle.src_lane_id:
                self.__add_edge_by_vtx(v, u, EdgeType.TYPE_3)

    def finish_execute(self, v: Vertex):
        key = (v.vehicle.id, v.cz_id)
        if key not in self.__V or id(self.__V[key]) != id(v):
            raise Exception("[TimingConflictGraph.finish_execute] supplied vertex does not belongs to this graph")
        v.state = VertexState.EXECUTED

    def reset_vertices_state(self) -> None:
        for v in self.__V.values():
            v.state = VertexState.NON_EXECUTED
        for key, e in list(self.__E.items()):
            if e.type == EdgeType.TYPE_3:
                e.v_from.out_edges.remove(e)
                e.v_to.in_edges.remove(e)
                del self.__E[key]

    def add_unsure_type3_edges(self) -> None:
        for v1 in self.__V.values():
            for v2 in self.__V.values():
                if v1.id != v2.id and v1.cz_id == v2.cz_id and (v1.id, v2.id) not in self.__E and (v2.id, v1.id) not in self.__E:
                    self.__add_edge_by_vtx(v1, v2, EdgeType.TYPE_3)
                    self.__add_edge_by_vtx(v2, v1, EdgeType.TYPE_3)

    @property
    def V(self) -> Iterable[Vertex]:
        return self.__V.values()
    
    @property
    def E(self) -> Iterable[Edge]:
        return self.__E.values()

    def get_v_by_idx(self, vehicle: Vehicle, cz_id: str) -> Vertex:
        return self.__V[(vehicle.id, cz_id)]

    def get_e_by_v_pair(self, v_from: Vertex, v_to: Vertex) -> Edge:
        return self.__E[(v_from.id, v_to.id)]

    def __add_vertex(self, vehicle: Vehicle, cz_id: str) -> None:
        if (vehicle.id, cz_id) not in self.__V:
            v = Vertex(len(self.__V), vehicle, cz_id)
            self.__V[(vehicle.id, cz_id)] = v

    def __add_edge_by_idx(self, veh_from: Vehicle, cz_id_from: str, veh_to: Vehicle, cz_id_to: str, type: EdgeType) -> None:
        v_from = self.__V[(veh_from.id, cz_id_from)]
        v_to = self.__V[(veh_to.id, cz_id_to)]
        self.__add_edge_by_vtx(v_from, v_to, type)

    def __add_edge_by_vtx(self, v_from: Vertex, v_to: Vertex, type: EdgeType) -> None:
        if (v_from.id, v_to.id) not in self.__E:
            edge = Edge(len(self.__E), v_from, v_to, type)
            self.__E[(v_from.id, v_to.id)] = edge
            v_from.add_out_edge(edge)
            v_to.add_in_edge(edge)

