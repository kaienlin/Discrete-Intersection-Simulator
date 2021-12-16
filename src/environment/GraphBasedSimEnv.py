import gym
from gym import spaces
import copy
from typing import Iterable, Set, Dict, List, Tuple
from functools import lru_cache

from simulator.intersection import Intersection
from simulator.simulation import Simulator
from simulator.vehicle import Vehicle, VehicleState

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
           pair (src, dst) where:
                + src can be a source lane or a conflict zone
                + dst can be a conflict zone or "$" (terminal)

        2. the number of vehicles queued in each source lane, which is
           represented as a n-tuple where n is the number of source lanes.
           The granularity of the queue size can be specified to control
           the size of the state space.
    
    **ACTION**
    The action space consists of the following elements:
        1. a source lane or a conflict zone where the vehicle (if exists) in
           the position is allowed to move to its destination in the previously
           returned state.

    **COST**
    The cost is a non-negative integer the waiting time increased between the
    previous and the current state.
    '''
    DEADLOCK_PENALTY = 1000000000
    def __init__(self, sim: Simulator, queue_size_scale: Tuple[int] = (1,)):
        super(GraphBasedSimEnv, self).__init__()
        self.sim = sim
        self.queue_size_scale = queue_size_scale

        # The sorted ids are for state encoding
        self.sorted_src_lane_ids = sorted(self.sim.intersection.src_lanes.keys())
        self.sorted_cz_ids = sorted(self.sim.intersection.conflict_zones)
        
        # compute the size of the state space and the action space
        self.state_space_size = self.__create_state_encoding(sim.intersection)
        self.action_space_size = self.__create_action_encoding(sim.intersection)

        # For following gym interface
        self.observation_space = spaces.Discrete(self.state_space_size)
        self.action_space = spaces.Discrete(self.action_space_size)

        # simulation-related object for calculating cost
        self.prev_timestamp: int = 0
        self.prev_vehicles = None
        self.prev_idle_veh: Set[str] = set()

    def reset(self):
        '''
        reset the environment and return a initial state
        '''
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

            # loop until reaching a state containing at least one waiting vehicles or terminal
            if len([veh.id for veh in vehicles if veh.state == VehicleState.WAITING]) > 0 or self.sim.status != "RUNNING":
                break

        next_state = self.encode_state(vehicles)
        terminal = self.sim.status != "RUNNING"
        if terminal and self.sim.status == "DEADLOCK":
            waiting_time_sum += self.DEADLOCK_PENALTY
        return next_state, waiting_time_sum, terminal, {}

    def __calc_queue_size_scale(self, queue_size: int):
        for idx, scale in enumerate(self.queue_size_scale):
            if queue_size < scale:
                return idx
        return len(self.queue_size_scale)

    def __create_action_encoding(self, intersection: Intersection) -> int:
        return len(intersection.conflict_zones) + len(intersection.src_lanes) + 1

    def __create_state_encoding(self, intersection: Intersection) -> int:
        trans_per_cz_id: Dict[str, Set[str]] = {cz_id: set() for cz_id in intersection.conflict_zones}
        for traj in intersection.trajectories:
            for idx, cz_1 in enumerate(traj[:-1]):
                cz_2 = traj[idx + 1]
                trans_per_cz_id[cz_1].add(cz_2)
            trans_per_cz_id[traj[-1]].add("$")
        self.trans_per_cz_id: Dict[str, List[str]] = {k: list(s) for k, s in trans_per_cz_id.items()}
        self.trans_per_src_lane: Dict[str, List[str]] = {k: list(s) for k, s in self.sim.intersection.src_lanes.items()}

        # number of states of vehicle position
        n_states = 1
        for associated_czs in intersection.src_lanes.values():
            n_states *= 2 * (len(associated_czs) + 1)
        for cz_ids in trans_per_cz_id.values():
            n_states *= len(cz_ids) + 1
        
        # number of (not left) vehicles in each source lane
        n_states *= (len(self.queue_size_scale) + 1) ** len(self.sorted_src_lane_ids)

        return n_states

    def is_invalid_state(self, state: int):
        state = self.decode_state(state)
        D = dict()
        for pos in state["vehicle_positions"]:
            D[(pos["type"], pos["id"])] = (pos["waiting"], pos["next_pos"])
        for (_, _id), (is_waiting, next_pos) in D.items():
            if is_waiting and ("cz", next_pos) in D:
                return True
        return False

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
        num_src_lane = len(self.sorted_src_lane_ids)
        num_cz = len(self.sorted_cz_ids)
        if 1 <= action <= num_src_lane:
            veh = self.sim.get_waiting_veh_of_src_lane(self.sorted_src_lane_ids[action - 1])
            return "" if veh is None else veh.id
        elif num_src_lane + 1 <= action <= num_cz + num_src_lane:
            veh = self.sim.get_waiting_veh_on_cz(self.sorted_cz_ids[action - num_src_lane - 1])
            return "" if veh is None else veh.id

    @lru_cache(maxsize=128)
    def decode_action(self, action: int) -> Dict:
        if action == 0:
            return {}
        num_src_lane = len(self.sorted_src_lane_ids)
        num_cz = len(self.sorted_src_lane_ids)
        if 1 <= action <= num_src_lane:
            return {"type": "src", "pos": f"{self.sorted_src_lane_ids[action - 1]}"}
        elif num_src_lane + 1 <= action <= num_cz + num_src_lane:
            return {"type": "cz", "pos": f"{self.sorted_cz_ids[action - num_src_lane - 1]}"}

    def encode_state(self, vehicles: Iterable[Vehicle]) -> int:
        queue_size_per_src_lane = {src_lane_id: 0 for src_lane_id in self.sim.intersection.src_lanes}
        cz_state = {cz_id: 0 for cz_id in self.sim.intersection.conflict_zones}
        src_lane_state = {src_lane_id: 0 for src_lane_id in self.sim.intersection.src_lanes}
        for veh in vehicles:
            if veh.idx_on_traj == -1 and veh.state == VehicleState.BLOCKED:
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

        for src_lane_id in self.sorted_src_lane_ids:
            trans = self.trans_per_src_lane[src_lane_id]
            state = state * (len(trans) + 1) + src_lane_state[src_lane_id]

        for src_lane_id in self.sorted_src_lane_ids:
            queue_size = queue_size_per_src_lane[src_lane_id]
            scale = self.__calc_queue_size_scale(queue_size)
            state = state * (len(self.queue_size_scale) + 1) + scale

        return state

    @lru_cache(maxsize=1024)
    def decode_state(self, state: int) -> Dict:
        res: Dict = {
            "queue_size_per_src_lane": {},
            "vehicle_positions": []
        }

        for src_lane_id in self.sorted_src_lane_ids[::-1]:
            scale = state % (len(self.queue_size_scale) + 1)
            state //= (len(self.queue_size_scale) + 1)
            queue_size = 0 if scale == 0 else self.queue_size_scale[scale-1]
            res["queue_size_per_src_lane"][src_lane_id] = queue_size

        for src_lane_id in self.sorted_src_lane_ids[::-1]:
            trans = self.trans_per_src_lane[src_lane_id]
            lane_state = state % (len(trans) + 1)
            state //= len(trans) + 1
            if lane_state > 0:
                res["vehicle_positions"].append({
                    "id": src_lane_id,
                    "type": "src",
                    "waiting": True,
                    "next_pos": trans[lane_state - 1]
                })
            
        for cz_id in self.sorted_cz_ids[::-1]:
            trans = self.trans_per_cz_id[cz_id]
            cz_state = state % (2 * (len(trans) + 1))
            state //= 2 * (len(trans) + 1)
            is_waiting = cz_state % 2
            cz_state //= 2
            if cz_state > 0:
                res["vehicle_positions"].append({
                    "id": cz_id,
                    "type": "cz",
                    "waiting": is_waiting,
                    "next_pos": trans[cz_state - 1]
                })
            
        return res  

