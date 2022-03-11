from .tcg import TimingConflictGraph, Vertex, Edge, VertexState, EdgeType
from .intersection import Intersection
from .vehicle import Vehicle, VehicleState

from typing import Iterable, Tuple, List, Dict
import heapq
import enum
import json

class EventType(enum.Enum):
    END_MOVING = enum.auto()
    START_WAITING = enum.auto()
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
            heapq.heappush(self.__heap, EventQueueItem(t, type, v))


class Simulator:
    def __init__(
        self,
        intersection: Intersection,
    ):
        self.__intersection: Intersection = intersection
        
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

    def dump_traffic(self, path) -> None:
        vehicle_dicts = []
        for veh in self.__vehicles.values():
            vehicle_dicts.append(veh.asdict())
        json.dump(vehicle_dicts, open(path, "w"), indent=2, sort_keys=True)

    def load_traffic(self, path) -> None:
        vehicle_dicts = json.load(open(path, "r"))
        for veh_dict in vehicle_dicts:
            self.add_vehicle(
                veh_dict["id"],
                veh_dict["earliest_arrival_time"],
                tuple(veh_dict["trajectory"]),
                veh_dict["src_lane_id"],
                veh_dict["dst_lane_id"],
                veh_dict["vertex_passing_time"]
            )

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
        if not self.__ready_condition(v):
            print(v.vehicle.state)
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
    def intersection(self) -> Intersection:
        return self.__intersection

    @property
    def vehicles(self) -> List[Vehicle]:
        return list(self.__vehicles.values())
        
    @property
    def status(self) -> str:
        return self.__status

    @property
    def timestamp(self) -> int:
        return self.__timestamp

    @property
    def TCG(self) -> TimingConflictGraph:
        return self.__TCG

    def print_TCG(self) -> None:
        self.__TCG.print()

    def get_waiting_veh_of_src_lane(self, src_lane_id: str):
        for veh in self.__vehicles.values():
            if veh.idx_on_traj == -1 and veh.src_lane_id == src_lane_id and veh.state == VehicleState.WAITING:
                return veh
        return None

    def get_waiting_veh_on_cz(self, cz_id: str):
        for veh in self.__vehicles.values():
            cur_cz_id = veh.get_cur_cz()
            if cur_cz_id == cz_id and veh.state == VehicleState.WAITING:
                return veh
        return None

    def is_frontmost_vehicle(self, vehicle: Vehicle) -> bool:
        if vehicle.get_cur_cz() != "^" or vehicle.state == VehicleState.NOT_ARRIVED:
            return False
        cz_id: str = vehicle.get_next_cz()
        v: Vertex = self.__TCG.get_v_by_idx(vehicle, cz_id)
        for in_e in v.in_edges:
            if in_e.type == EdgeType.TYPE_2 and in_e.v_from.state == VertexState.NON_EXECUTED:
                return False
        return True
            
    def run(self) -> None:
        self.__status = "RUNNING"
        self.__TCG = TimingConflictGraph(set(self.__vehicles.values()), self.__intersection)
        self.__enqueue_front_vertices()
    
    def reset_simulation(self) -> None:
        self.__status = "INITIALIZED"
        self.__timestamp = -1
        self.__event_queue.clear()
        self.__TCG.reset_vertices_state()
        for veh in self.__vehicles.values():
            veh.reset()

    def handle_event_queue(self) -> None:
        while not self.__event_queue.empty() and self.__event_queue.top().time == self.__timestamp:
            ev = self.__event_queue.top()
            self.__event_queue.pop()
            if ev.type == EventType.END_MOVING:
                if ev.vertex.vehicle.on_last_cz():
                    ev.vertex.vehicle.set_state(VehicleState.WAITING)
                elif self.__next_not_blocked(ev.vertex):
                    ev.vertex.vehicle.set_state(VehicleState.BLOCKED)
                    next_v = self.__TCG.get_v_by_idx(ev.vertex.vehicle, ev.vertex.vehicle.get_next_cz())
                    avail_time = self.__compute_earliest_entering_time(next_v)
                    self.__event_queue.push(avail_time, EventType.START_WAITING, next_v)
                else:
                    ev.vertex.vehicle.set_state(VehicleState.BLOCKED)
            elif ev.type == EventType.ARRIVAL:
                if self.__not_blocked(ev.vertex):
                    ev.vertex.vehicle.set_state(VehicleState.WAITING)
                else:
                    ev.vertex.vehicle.set_state(VehicleState.BLOCKED)
            elif ev.type == EventType.START_WAITING:
                if self.__not_blocked(ev.vertex):
                    ev.vertex.vehicle.set_state(VehicleState.WAITING)

    def simulation_step_report(self) -> Tuple[int, Iterable[Vehicle]]:
        if any([veh.state == VehicleState.WAITING for veh in self.__vehicles.values()]):
            if not self.__prev_moved:
                self.__timestamp += 1
            else:
                self.__prev_moved = True
        elif not self.__event_queue.empty():
            assert self.__timestamp != self.__event_queue.top().time
            self.__timestamp = min(self.__event_queue.top().time, self.__timestamp + 1)
        else:
            if any([veh.state != VehicleState.LEFT for veh in self.__vehicles.values()]):
                self.__status = "DEADLOCK"
            else:
                self.__status = "TERMINATED"
            return (self.__timestamp, list(self.__vehicles.values()))

        self.handle_event_queue()

        return (self.__timestamp, list(self.__vehicles.values()))

    def __release_cz(self, vertex: Vertex):
        for out_e in vertex.out_edges:
            if out_e.type == EdgeType.TYPE_1:
                continue
            child = out_e.v_to
            if child.vehicle.state == VehicleState.BLOCKED and self.__ready_condition(child):
                avail_time = self.__compute_earliest_entering_time(child)
                self.__event_queue.push(avail_time, EventType.START_WAITING, child)

    def __block_cz(self, vertex: Vertex):
        for out_e in vertex.out_edges:
            if out_e.type == EdgeType.TYPE_1:
                continue
            child = out_e.v_to
            if child.vehicle.state == VehicleState.WAITING and child.vehicle.get_next_cz() == vertex.cz_id \
               and not self.__ready_condition(child):
                child.vehicle.set_state(VehicleState.BLOCKED)

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

        next_cz = vehicle.get_next_cz()
        next_vertex = None
        if next_cz != "$":
            next_vertex = self.__TCG.get_v_by_idx(vehicle, next_cz)
            next_vertex.entering_time = self.__timestamp

        if vehicle.idx_on_traj != -1:
            cur_vertex = self.__TCG.get_v_by_idx(vehicle, vehicle.get_cur_cz())
            self.__TCG.finish_execute(cur_vertex)
            self.__release_cz(cur_vertex)
            self.handle_event_queue()

        vehicle.move_to_next_cz()

        if next_cz == "$":
            vehicle.set_state(VehicleState.LEFT)
            return

        vehicle.set_state(VehicleState.MOVING)
        
        self.__event_queue.push(next_vertex.entering_time + next_vertex.get_consumed_time(), EventType.END_MOVING, next_vertex)
        self.__TCG.start_execute(next_vertex)
        self.__block_cz(next_vertex)
