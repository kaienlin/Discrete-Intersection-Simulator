import gym
from gym import spaces
import math
import copy
from typing import Iterable, Set, Dict, List, Tuple

from simulator.intersection import Intersection
from simulator.simulation import Simulator
from simulator.vehicle import Vehicle, VehicleState

MAX_WAITING_TIME_SUM = 1000000000
MAX_VEHICLE_NUM = 8
DEADLOCK_PENALTY = 1000000000

def gen_int_partition(n: int, k: int):
    res: Set[Tuple[int]] = set()
    def __gen(sum: int, L: List[int]) -> None:
        if len(L) == k:
            res.add(tuple(L))
            return
        for i in range(0, n - sum + 1):
            L.append(i)
            __gen(sum + i, L)
            L.pop()
    __gen(0, [])
    return res             
        
 
class GraphBasedSimEnv(gym.Env):
    '''
    **STATE**
    The state space consists of the following elements:
        1. all possible combinations of "transitions". A transition is a
           pair (src, dst) where src, dst can be conflict zones or src/dst lanes.

        2. the number of vehicles that have not left in each source lane.
    '''
    def __init__(self, sim: Simulator):
        super(GraphBasedSimEnv, self).__init__()
        self.sim = sim
        self.sorted_src_lane_ids = sorted(self.sim.intersection.src_lanes.keys())
        self.sorted_cz_ids = sorted(self.sim.intersection.conflict_zones)
        
        self.state_space_size = self.__create_state_enc(sim.intersection)
        self.action_space_size = self.__create_action_enc(sim.intersection)

        self.observation_space = spaces.Discrete(self.state_space_size)
        self.action_space = spaces.Discrete(self.action_space_size)

        self.prev_timestamp: int = 0
        self.prev_vehicles = None
        self.prev_idle_veh: Set[str] = set()

    def reset(self):
        self.sim.reset_simulation()
        self.sim.run()
        timestamp, vehicles = self.sim.simulation_step_report()
        self.prev_timestamp = timestamp
        self.prev_vehicles = copy.deepcopy(vehicles)
        self.prev_idle_veh = set([veh.id for veh in vehicles if self.__is_idle_state(veh.state)])
        return self.encode_state(vehicles)

    def print_state(self, t, vehicles):
        print(f"State at time {t}")
        for veh in vehicles:
            print(f"  - {veh.id}: {veh.get_cur_cz()} -> {veh.get_next_cz()} {veh.state}")

    def render(self):
        self.print_state(self.prev_timestamp, self.prev_vehicles)

    def __is_idle_state(self, state: VehicleState) -> bool:
        return state == VehicleState.WAITING or state == VehicleState.BLOCKED

    def step(self, action: int):
        veh_id = self.decode_action_to_veh_id(action)
        self.sim.simulation_step_act(veh_id)
        waiting_time_sum = 0
        vehicles = []
        while True:
            timestamp, vehicles = self.sim.simulation_step_report()
            num_waiting = len(self.prev_idle_veh) - (1 if veh_id in self.prev_idle_veh else 0)
            waiting_time_sum += (timestamp - self.prev_timestamp) * num_waiting
            self.prev_timestamp = timestamp
            self.prev_vehicles = copy.deepcopy(vehicles)
            self.prev_idle_veh = set([veh.id for veh in vehicles if self.__is_idle_state(veh.state)])
            if len([veh.id for veh in vehicles if veh.state == VehicleState.WAITING]) > 0 or self.sim.status != "RUNNING":
                break
        next_state = self.encode_state(vehicles)
        terminal = self.sim.status != "RUNNING"
        if terminal and self.sim.status == "DEADLOCK":
            waiting_time_sum += DEADLOCK_PENALTY
        return next_state, waiting_time_sum, terminal, {}

    def __create_action_enc(self, intersection: Intersection) -> int:
        return len(intersection.conflict_zones) + len(intersection.src_lanes) + 1

    def __create_state_enc(self, intersection: Intersection) -> int:
        trans_per_cz_id: Dict[str, Set[str]] = {cz_id: set() for cz_id in intersection.conflict_zones}
        for traj in intersection.trajectories:
            for idx, cz_1 in enumerate(traj[:-1]):
                cz_2 = traj[idx + 1]
                trans_per_cz_id[cz_1].add(cz_2)
            trans_per_cz_id[traj[-1]].add("$")
        self.trans_per_cz_id: Dict[str, List[str]]= {k: list(s) for k, s in trans_per_cz_id.items()}
        self.trans_per_src_lane: Dict[str, List[str]] = {k: list(s) for k, s in self.sim.intersection.src_lanes.items()}

        # number of states of vehicle position
        n_states = 1
        for associated_czs in intersection.src_lanes.values():
            #n_states *= len(associated_czs) + 1
            n_states *= 2 * (len(associated_czs) + 1)
        for cz_ids in trans_per_cz_id.values():
            n_states *= len(cz_ids) + 1
        
        # number of (not left) vehicles in each source lane
        k = len(self.sim.intersection.src_lanes)
        self.queue_sizes_max_no = math.comb(MAX_VEHICLE_NUM + k, k)
        n_states *= self.queue_sizes_max_no

        self.queue_sizes_no_map: Dict[Tuple[int], int] = dict()
        self.queue_sizes_no_inv: Dict[int, Tuple[int]] = dict()
        partitions = gen_int_partition(MAX_VEHICLE_NUM, len(self.sim.intersection.src_lanes))

        for i, p in enumerate(partitions):
            self.queue_sizes_no_map[p] = i
            self.queue_sizes_no_inv[i] = p

        return n_states

    def encode_action(self, _id: str, is_cz: bool) -> int:
        if _id == "":
            return 0
        num_src_lane = len(self.sorted_src_lane_ids)
        if is_cz:
            return self.sorted_cz_ids.index(_id) + num_src_lane + 1
        else:
            return self.sorted_src_lane_ids.index(_id) + 1
        
    def decode_action_to_veh_id(self, action: int) -> str:
        if action == 0:
            return ""
        num_src_lane = len(self.sim.intersection.src_lanes)
        num_cz = len(self.sim.intersection.conflict_zones)
        if 1 <= action <= num_src_lane:
            veh = self.sim.get_waiting_veh_of_src_lane(self.sorted_src_lane_ids[action - 1])
            return "" if veh is None else veh.id
        elif num_src_lane + 1 <= action <= num_cz + num_src_lane:
            veh = self.sim.get_waiting_veh_on_cz(self.sorted_cz_ids[action - num_src_lane - 1])
            return "" if veh is None else veh.id

    def decode_action(self, action: int) -> str:
        if action == 0:
            return ""
        num_src_lane = len(self.sim.intersection.src_lanes)
        num_cz = len(self.sim.intersection.conflict_zones)
        if 1 <= action <= num_src_lane:
            return {"type": "src", "pos": f"{self.sorted_src_lane_ids[action - 1]}"}
        elif num_src_lane + 1 <= action <= num_cz + num_src_lane:
            return {"type": "cz", "pos": f"{self.sorted_cz_ids[action - num_src_lane - 1]}"}

    def encode_state(self, vehicles: Iterable[Vehicle]) -> int:
        queue_size_per_src_lane = {src_lane_id: 0 for src_lane_id in self.sim.intersection.src_lanes}
        cz_state = {cz_id: 0 for cz_id in self.sim.intersection.conflict_zones}
        src_lane_state = {src_lane_id: 0 for src_lane_id in self.sim.intersection.src_lanes}
        for veh in vehicles:
            if veh.idx_on_traj == -1 and veh.state != VehicleState.WAITING:
                queue_size_per_src_lane[veh.src_lane_id] += 1
            elif veh.state != VehicleState.LEFT:
                veh_pos = veh.get_cur_cz()
                next_veh_pos = veh.get_next_cz()
                if veh_pos == "^":
                    if veh.state == VehicleState.WAITING:
                        src_lane_state[veh.src_lane_id] = self.trans_per_src_lane[veh.src_lane_id].index(next_veh_pos) + 1
                elif veh_pos != "$":
                    cz_state[veh_pos] = self.trans_per_cz_id[veh_pos].index(next_veh_pos) + 1
                    cz_state[veh_pos] *= 2
                    if veh.state == VehicleState.WAITING:
                        cz_state[veh_pos] += 1
        
        state: int = 0

        for cz_id in self.sorted_cz_ids:
            trans = self.trans_per_cz_id[cz_id]
            state = state * 2 * (len(trans) + 1) + cz_state[cz_id]
            #state = state * (len(trans) + 1) + cz_state[cz_id]

        for src_lane_id in self.sorted_src_lane_ids:
            trans = self.trans_per_src_lane[src_lane_id]
            state = state * (len(trans) + 1) + src_lane_state[src_lane_id]

        queue_size_list = []
        for src_lane_id in self.sorted_src_lane_ids:
            queue_size = queue_size_per_src_lane[src_lane_id]
            queue_size_list.append(queue_size)
        state = state * self.queue_sizes_max_no + self.queue_sizes_no_map[tuple(queue_size_list)]

        return state

    def decode_state(self, state: int) -> Dict:
        res: Dict = {
            "queue_size_per_src_lane": {},
            "vehicle_positions": {}
        }

        queue_size_tuple = self.queue_sizes_no_inv[state % self.queue_sizes_max_no]
        state //= self.queue_sizes_max_no
        for i, src_lane_id in enumerate(self.sorted_src_lane_ids):
            res["queue_size_per_src_lane"][src_lane_id] = queue_size_tuple[i]

        for src_lane_id in self.sorted_src_lane_ids[::-1]:
            trans = self.trans_per_src_lane[src_lane_id]
            lane_state = state % (len(trans) + 1)
            state //= len(trans) + 1
            if lane_state > 0:
                res["vehicle_positions"][f"{src_lane_id}"] = {
                    "type": "src",
                    "waiting": True,
                    "next_cz": trans[lane_state - 1]
                }
            
        for cz_id in self.sorted_cz_ids[::-1]:
            trans = self.trans_per_cz_id[cz_id]
            cz_state = state % (2 * (len(trans) + 1))
            state //= 2 * (len(trans) + 1)
            #cz_state = state % (len(trans) + 1)
            #state //= len(trans) + 1
            is_waiting = cz_state % 2
            cz_state //= 2
            if cz_state > 0:
                res["vehicle_positions"][cz_id] = {
                    "type": "cz",
                    "waiting": is_waiting,
                    "next_pos": trans[cz_state - 1]
                }
            
        return res  

