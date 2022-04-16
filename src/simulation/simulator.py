from typing import Optional, Tuple, List, Dict, Union, Set
import enum
import json
import random

from .tcg import TimingConflictGraph, Vertex, VertexState
from .intersection import Intersection
from .vehicle import Vehicle, VehicleState


class SimulatorStatus(enum.Enum):
    INITIALIZED = enum.auto()
    RUNNING = enum.auto()
    TERMINATED = enum.auto()
    DEADLOCK = enum.auto()


class DeadlockException(Exception):
    pass


class Simulator:
    def __init__(
        self,
        intersection: Intersection,
        disturbance_prob: Optional[float] = None,
    ):
        self._intersection: Intersection = intersection
        self.disturbance_prob: Union[None, float] = disturbance_prob

        self._vehicles: Dict[str, Vehicle] = {}
        self._status: str = SimulatorStatus.INITIALIZED
        self._timestamp: int = -1
        self._TCG: TimingConflictGraph = TimingConflictGraph(
            set(self._vehicles.values()), intersection)

        self._executing_vertices: Set[Vertex] = set()
        self._non_executed_vertices: Set[Vertex] = set()

    @property
    def intersection(self) -> Intersection:
        return self._intersection

    @property
    def vehicles(self) -> List[Vehicle]:
        return list(self._vehicles.values())

    @property
    def status(self) -> str:
        return self._status

    @property
    def timestamp(self) -> int:
        return self._timestamp

    @property
    def TCG(self) -> TimingConflictGraph:
        return self._TCG

    def add_vehicle(
        self,
        _id: str,
        arrival_time: int,
        trajectory: Tuple[str],
        src_lane_id: str,
        dst_lane_id: str,
        vertex_passing_time: int = 10
    ):
        if self._status == SimulatorStatus.RUNNING:
            raise Exception("cannot add vehicle when the simulator is running")
        if _id in self._vehicles:
            raise Exception("_id has been used")
        if arrival_time < 0:
            raise Exception("negative arrival_time")
        if len(trajectory) == 0:
            raise Exception("empty trajectory")
        if src_lane_id not in self._intersection.src_lanes:
            raise Exception("src_lane_id not in intersection")
        if dst_lane_id not in self._intersection.dst_lanes:
            raise Exception("dst_lane_id not in intersection")
        if trajectory[0] not in self._intersection.src_lanes[src_lane_id]:
            raise Exception("the first CZ of trajectory does not belongs to src_lane_id")
        if trajectory[-1] not in self._intersection.dst_lanes[dst_lane_id]:
            raise Exception("the last CZ of trajectory does not belongs to dst_lane_id")
        if vertex_passing_time < 0:
            raise Exception("negative vertex_passing_time")

        vehicle = Vehicle(
            _id,
            arrival_time,
            trajectory,
            src_lane_id,
            dst_lane_id,
            vertex_passing_time
        )
        self._vehicles[vehicle.id] = vehicle

    def remove_vehicle(self, vehicle_id: str) -> None:
        vehicle: Vehicle = self._vehicles[vehicle_id]
        for cz_id in vehicle.trajectory:
            v = self._TCG.get_vertex_by_vehicle_cz_pair(vehicle, cz_id)
            self._TCG.remove_vertex(v)
        del self._vehicles[vehicle.id]

    def dump_traffic(self, path) -> None:
        vehicle_dicts = []
        for veh in self._vehicles.values():
            vehicle_dicts.append(veh.asdict())
        with open(path, "wt", encoding="utf-8") as f:
            json.dump(vehicle_dicts, f, indent=2, sort_keys=True)

    def load_traffic(self, path) -> None:
        with open(path, "rt", encoding="utf-8") as f:
            vehicle_dicts = json.load(f)
        for veh_dict in vehicle_dicts:
            self.add_vehicle(
                veh_dict["id"],
                veh_dict["earliest_arrival_time"],
                tuple(veh_dict["trajectory"]),
                veh_dict["src_lane_id"],
                veh_dict["dst_lane_id"],
                veh_dict["vertex_passing_time"]
            )

    def print_TCG(self) -> None:
        self._TCG.print()

    def run(self) -> None:
        self._TCG = TimingConflictGraph(set(self._vehicles.values()), self._intersection)
        self.reset()
        self._status = SimulatorStatus.RUNNING

    def reset(self) -> None:
        self._status = SimulatorStatus.INITIALIZED
        self._timestamp = -1
        self._TCG.reset_vertices_state()
        for vehicle in self._vehicles.values():
            vehicle.reset()
        for vertex in self._TCG.V:
            self._non_executed_vertices.add(vertex)
        self.step(None)

    def _check_deadlock_dfs(self, vertex: Vertex, color: List[int]) -> bool:
        color[vertex.id] = 1
        for out_edge in vertex.out_edges:
            if not out_edge.decided:
                continue
            if color[out_edge.v_to.id] == 0:
                if self._check_deadlock_dfs(out_edge.v_to, color):
                    return True
            elif color[out_edge.v_to.id] == 1:
                return True
        color[vertex.id] = 2
        return False

    def check_deadlock(self) -> bool:
        color = [0 for _ in self._TCG.V]
        for vertex in self._TCG.V:
            if color[vertex.id] == 0:
                if self._check_deadlock_dfs(vertex, color):
                    return True
        return False

    def _compute_earliest_entering_time(self, vertex: Vertex) -> None:
        res: int = self._timestamp

        if vertex.cz_id == vertex.vehicle.trajectory[0]:
            res = max(res, vertex.vehicle.earliest_arrival_time)

        for in_e in vertex.in_edges:
            if not in_e.decided:
                continue

            parent: Vertex = in_e.v_from

            if parent.earliest_entering_time is None:
                self._compute_earliest_entering_time(parent)

            res = max(res, parent.earliest_entering_time \
                           + parent.passing_time + in_e.waiting_time)

        vertex.earliest_entering_time = res

    def _update_all_earliest_entering_time(self) -> None:
        if self.check_deadlock():
            raise DeadlockException()

        for vertex in self._non_executed_vertices:
            vertex.earliest_entering_time = None

        for vertex in self._non_executed_vertices:
            self._compute_earliest_entering_time(vertex)

    def get_executable_vertices(self) -> Dict[str, Vertex]:
        res: Dict[str, Vertex] = {}
        for vertex in self._non_executed_vertices:
            if vertex.earliest_entering_time == self._timestamp:
                res[vertex.vehicle.id] = vertex
        return res

    def step(self, moved_vehicle_id: Optional[str]):
        if len(self._non_executed_vertices) == 0:
            self._status = SimulatorStatus.TERMINATED
            return

        executable_vertices: Dict[str, Vertex] = self.get_executable_vertices()
        vertex_to_be_executed: Union[None, Vertex] = executable_vertices.get(moved_vehicle_id, None)

        if vertex_to_be_executed is not None:
            # Remove Type-3 edge, Add Type-4 edge, set vertex.state = EXECUTING
            self._TCG.start_execute(vertex_to_be_executed)

            # Update vertex information
            vertex_to_be_executed.entering_time = self._timestamp
            vertex_to_be_executed.earliest_entering_time = self._timestamp
            self._non_executed_vertices.remove(vertex_to_be_executed)
            self._executing_vertices.add(vertex_to_be_executed)

            # Update vehicle information
            vertex_to_be_executed.vehicle.move_to_next_cz()
            vertex_to_be_executed.vehicle.set_state(VehicleState.MOVING)

        # If there is no vehicle moved or there is no more vehicles can be moved
        if vertex_to_be_executed is None or len(executable_vertices) == 1:
            # move to the next time step
            self._timestamp += 1

        # finish executing
        for vertex in list(self._executing_vertices):
            if self._timestamp == vertex.entering_time + vertex.passing_time:
                self._executing_vertices.remove(vertex)
                vertex.state = VertexState.EXECUTED
                vertex.vehicle.set_state(VehicleState.BLOCKED)

        try:
            self._update_all_earliest_entering_time()

            for vehicle in self._vehicles.values():
                if vehicle.state == VehicleState.NOT_ARRIVED \
                   and vehicle.earliest_arrival_time == self._timestamp:
                    vehicle.set_state(VehicleState.BLOCKED)

            for vertex in self._non_executed_vertices:
                if vertex.earliest_entering_time == self._timestamp:
                    vertex.vehicle.set_state(VehicleState.READY)
        except DeadlockException:
            self.status = SimulatorStatus.DEADLOCK

    def observe(self):
        res = {
            "vehicles": self._vehicles.values(),
            "time": self._timestamp
        }
        return res
