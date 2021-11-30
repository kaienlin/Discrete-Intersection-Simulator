from .tcg import TimingConflictGraph, Vertex, Edge, VertexState, EdgeType
from .intersection import Intersection
from .vehicle import Vehicle, VehicleState

from typing import Iterable, Tuple, List, Dict
import heapq
import enum

class EventType(enum.Enum):
    END_MOVING = enum.auto()
    ARRIVAL = enum.auto()


class EventQueueItem:
    def __init__(self, t: int, type: EventType, v: Vertex):
        self.time = t
        self.type = type
        self.vertex = v

    def __lt__(self, other) -> bool:
        return self.time < other.time


class VertexEventQueue:
    def __init__(self, initial_elements: List[EventQueueItem] = []):
        self.__heap: List[EventQueueItem] = initial_elements
        heapq.heapify(self.__heap)
    
    def clear(self) -> None:
        self.__heap = []
        heapq.heapify(self.__heap)
    
    def empty(self) -> bool:
        return len(self.__heap) == 0

    def top(self) -> EventQueueItem:
        return self.__heap[0]

    def pop(self) -> None:
        heapq.heappop(self.__heap)

    def push(self, t: int, type: EventType, v: Vertex):
        if EventQueueItem(t, type, v) not in self.__heap:
            #print(f"[push] {v.vehicle.id} {type}")
            heapq.heappush(self.__heap, EventQueueItem(t, type, v))


class Simulator:
    def __init__(
        self,
        intersection: Intersection,
    ):
        self.__intersection: Intersection = intersection
        self.__mode: str = "MANAGED"
        
        self.__vehicles: Dict[str, Vehicle] = dict()
        self.__status: str = "INITIALIZED"
        self.__timestamp: int = 0
        self.__TCG: TimingConflictGraph = TimingConflictGraph(set(self.__vehicles.values()), self.__intersection)
        self.__event_queue = VertexEventQueue()
        self.__prev_moved = False

    def add_vehicle(
        self,
        _id: str,
        arrival_time: int,
        trajectory: Tuple[str],
        src_lane_id: str,
        dst_lane_id: str,
        vertex_passing_time: int = 10
    ):
        if _id in self.__vehicles:
            raise Exception("[Simulator.add_vehicle] _id has been used")
        if arrival_time < 0:
            raise Exception("[Simulator.add_vehicle] negative arrival_time")
        if len(trajectory) == 0:
            raise Exception("[Simulator.add_vehicle] empty trajectory")
        if src_lane_id not in self.__intersection.src_lanes:
            raise Exception("[Simulator.add_vehicle] src_lane_id not in intersection")
        if dst_lane_id not in self.__intersection.dst_lanes:
            raise Exception("[Simulator.add_vehicle] dst_lane_id not in intersection")
        if trajectory[0] not in self.__intersection.src_lanes[src_lane_id]:
            raise Exception("[Simulator.add_vehicle] the first CZ of trajectory does not belongs to src_lane_id")
        if trajectory[-1] not in self.__intersection.dst_lanes[dst_lane_id]:
            raise Exception("[Simulator.add_vehicle] the last CZ of trajectory does not belongs to dst_lane_id")
        if vertex_passing_time < 0:
            raise Exception("[Simulator.add_vehicle] negative vertex_passing_time")

        vehicle = Vehicle(
            _id,
            arrival_time,
            trajectory,
            src_lane_id,
            dst_lane_id,
            vertex_passing_time
        )
        self.__vehicles[vehicle.id] = vehicle
        if self.__status == "RUNNING":
            self.__TCG.add_vehicle(vehicle)
            v = self.__TCG.get_v_by_idx(vehicle, vehicle.trajectory[0])
            self.__event_queue.push(max(self.__timestamp + 1, vehicle.earliest_arrival_time), EventType.ARRIVAL, v)

    def __ready_condition(self, v: Vertex) -> bool:
        if v.vehicle.state != VehicleState.WAITING and v.vehicle.state != VehicleState.BLOCKED:
            return False
        if v.state != VertexState.NON_EXECUTED:
            return False
        for in_e in v.in_edges:
            parent = in_e.v_from
            if in_e.type == EdgeType.TYPE_1 and parent.state != VertexState.EXECUTING:
                return False
            if in_e.type != EdgeType.TYPE_1 and parent.state != VertexState.EXECUTED:
                return False
        return True

    def __not_blocked(self, v: Vertex) -> bool:
        '''
        return true if v is not blocked by higher priority vehicles
        '''
        for in_e in v.in_edges:
            parent = in_e.v_from
            if in_e.type != EdgeType.TYPE_1 and parent.state != VertexState.EXECUTED:
                return False
        return True

    def __next_not_blocked(self, v: Vertex) -> bool:
        '''
        return true if the next CZ of v.vehicle is not blocked by higher priority vehicles
        '''
        next_v = self.__TCG.get_v_by_idx(v.vehicle, v.vehicle.trajectory[v.vehicle.idx_on_traj + 1])
        return self.__not_blocked(next_v)

    def __compute_earliest_entering_time(self, v: Vertex) -> int:
        '''
        compute the "earliest entering time" of a vertex
        for each parent p of vertex v:
            if (p, v) is an TYPE_1 edge, then p.state should be executing
            if (p, v) is an TYPE_2 or TYPE_3 edge, then p.state should be executed
        this is called "ready condition" temporarily
        '''
        assert(self.__ready_condition(v))
        res = self.__timestamp
        if v.cz_id == v.vehicle.trajectory[0]:
            res = max(res, v.vehicle.earliest_arrival_time)
        for in_e in v.in_edges:
            parent = in_e.v_from
            res = max(res, parent.entering_time + parent.passing_time + in_e.waiting_time)
            for out_e in parent.out_edges:
                if out_e.id != in_e.id and out_e.type == EdgeType.TYPE_1:
                    res = max(res, out_e.v_to.entering_time - out_e.waiting_time + in_e.waiting_time)
        return res

    def __enqueue_front_vertices(self) -> None:
        for vehicle in self.__vehicles.values():
            v = self.__TCG.get_v_by_idx(vehicle, vehicle.trajectory[0])
            self.__event_queue.push(vehicle.earliest_arrival_time, EventType.ARRIVAL, v)

    @property
    def status(self) -> str:
        return self.__status

    def print_TCG(self) -> None:
        self.__TCG.print()

    def run(self) -> None:
        self.__status = "RUNNING"
        self.__TCG = TimingConflictGraph(set(self.__vehicles.values()), self.__intersection)
        self.__enqueue_front_vertices()
    
    def reset_simulation(self) -> None:
        self.__timestamp = 0
        self.__event_queue.clear()
        self.__reset_vertices_state()

    def simulation_step_report(self) -> Iterable[Vehicle]:
        if any([veh.state == VehicleState.WAITING for veh in self.__vehicles.values()]):
            if not self.__prev_moved:
                self.__timestamp += 1
        elif not self.__event_queue.empty():
            self.__timestamp = self.__event_queue.top().time
        else:
            if any([veh.state != VehicleState.LEAVED for veh in self.__vehicles.values()]):
                self.__status = "DEADLOCK"
            else:
                self.__status = "TERMINATED"
            return (self.__timestamp, list(self.__vehicles.values()))

        while not self.__event_queue.empty() and self.__event_queue.top().time == self.__timestamp:
            ev = self.__event_queue.top()
            self.__event_queue.pop()
            #print(f"[report] {ev.vertex.vehicle.id} {ev.type}")
            if ev.type == EventType.END_MOVING:
                if ev.vertex.vehicle.on_last_cz():
                    ev.vertex.vehicle.set_state(VehicleState.LEAVED)
                    self.__TCG.finish_execute(ev.vertex)
                elif self.__next_not_blocked(ev.vertex):
                    ev.vertex.vehicle.set_state(VehicleState.WAITING)
                else:
                    ev.vertex.vehicle.set_state(VehicleState.BLOCKED)
            elif ev.type == EventType.ARRIVAL:
                if self.__not_blocked(ev.vertex):
                    ev.vertex.vehicle.set_state(VehicleState.WAITING)
                else:
                    ev.vertex.vehicle.set_state(VehicleState.BLOCKED)   

        return (self.__timestamp, list(self.__vehicles.values()))

    def simulation_step_act(self, allowed_veh_id: str) -> None:
        if allowed_veh_id == "":
            self.__prev_moved = False
            return

        if allowed_veh_id not in self.__vehicles \
           or self.__vehicles[allowed_veh_id].state != VehicleState.WAITING:
            self.__prev_moved = False
            return

        self.__prev_moved = True
        vehicle = self.__vehicles[allowed_veh_id]
        if vehicle.idx_on_traj != -1:
            cur_vertex = self.__TCG.get_v_by_idx(vehicle, vehicle.get_cur_cz())
            self.__TCG.finish_execute(cur_vertex)
            for out_e in cur_vertex.out_edges:
                if out_e.type == EdgeType.TYPE_1:
                    continue
                child = out_e.v_to
                if self.__ready_condition(child):
                    child.vehicle.set_state(VehicleState.WAITING)
        next_cz = vehicle.trajectory[vehicle.idx_on_traj + 1]
        next_vertex = self.__TCG.get_v_by_idx(vehicle, next_cz)

        vehicle.move_to_next_cz()
        vehicle.set_state(VehicleState.MOVING)
        next_vertex.entering_time = self.__timestamp
        self.__event_queue.push(next_vertex.entering_time + next_vertex.get_consumed_time(), EventType.END_MOVING, next_vertex)
        self.__TCG.start_execute(next_vertex)
                
