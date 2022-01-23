from collections import namedtuple
from typing import Tuple, List, Set
from tqdm import tqdm
import copy, math

from simulator.intersection import Intersection
from environment.position_based.BaseEnv import PositionBasedStateEnv
from utility import Digraph

class ProbabilisticEnv(PositionBasedStateEnv):
    def __init__(
        self,
        intersection: Intersection,
        queue_size_scale: Tuple[int] = (1,),
        traffic_density: float = 0.05,
        enable_reachability_analysis: bool = False
    ):
        super().__init__(
            intersection, 
            queue_size_scale=queue_size_scale,
        )

        self.traffic_density: float = traffic_density
        self.enable_reachability_analysis: bool = enable_reachability_analysis

        self.P: List[List[List[Tuple[float, int, int]]]] = [
            [None for a in range(self.action_space_size)]
            for s in range(self.state_space_size)
        ]

        if self.enable_reachability_analysis:
            self.reachable_states_without_op: List[Set[int]] = [None for s in range(self.state_space_size)]
            self.reachability_analysis()

    def get_higher_level_queue_size(self, queue_size: int):
        cur_level = self._discretize_queue_size(queue_size)
        if cur_level == len(self.queue_size_scale):
            return queue_size
        return self.queue_size_scale[cur_level]

    def get_lower_level_queue_size(self, queue_size: int):
        cur_level = self._discretize_queue_size(queue_size)
        if cur_level == 0 or cur_level == 1:
            return 0
        return self.queue_size_scale[cur_level - 2]

    def get_transitions(self, s: int, a: int):
        '''
        return a list of 3-tuple (prob, next_state, cost)
        '''
        # use cache
        if self.P[s][a] is not None:
            return self.P[s][a]

        if s == self.TERMINAL_STATE:
            return [(1.0, self.TERMINAL_STATE, 0)]
        elif self.deadlock_state_table[s]:
            return [(1.0, self.TERMINAL_STATE, self.DEADLOCK_COST)]

        res: List[Tuple[float, int, int]] = []
        origin_s_dec: PositionBasedStateEnv.DecodedState = self.decode_state(s)
        a_dec: PositionBasedStateEnv.DecodedAction = self.decode_action(a)

        cost: float = 0.0
        num_vehicles = 0
        for src_lane_state in origin_s_dec.src_lane_state.values():
            if src_lane_state.vehicle_state != "moving":
                cost += 1
            if src_lane_state.vehicle_state != "":
                num_vehicles += 1
            cost += src_lane_state.queue_size
            num_vehicles += src_lane_state.queue_size

        for cz_state in origin_s_dec.cz_state.values():
            if cz_state.vehicle_state != "moving":
                cost += 1
            if cz_state.vehicle_state != "":
                num_vehicles += 1

        def explore_cz(i: int, prob: float, sp: PositionBasedStateEnv.DecodedState):
            '''
            Explore the possible changes to the vehicle on a specific conflict zone
            '''
            if i == len(self.sorted_cz_ids):
                res.append((prob, self.encode_state(sp), cost))
                return

            cz_id: str = self.sorted_cz_ids[i]
            next_pos: str = origin_s_dec.cz_state[cz_id].next_position
            veh_state: str = origin_s_dec.cz_state[cz_id].vehicle_state

            VehicleStateTransition = namedtuple("VehicleStateTransition", ["src", "dst", "condition", "probability"])
            veh_state_transition_cfg: List[VehicleStateTransition] = [
                VehicleStateTransition(
                    src="moving",
                    dst=("waiting", "blocked"),
                    condition=lambda: next_pos == "$" or not sp.cz_state[next_pos].next_position,
                    probability=(0.06, 0.007)
                ),
                VehicleStateTransition(
                    src="moving",
                    dst=("blocked",),
                    condition=lambda: next_pos != "$" and sp.cz_state[next_pos].next_position,
                    probability=(0.067,)
                ),
                VehicleStateTransition(
                    src="blocked",
                    dst=("waiting",),
                    condition=lambda: next_pos == "$" or not sp.cz_state[next_pos].next_position,
                    probability=(0.8,)
                )
            ]

            if next_pos:
                satisfied_transition: List[VehicleStateTransition] = []
                for trans in veh_state_transition_cfg:
                    if trans.src == veh_state and trans.condition():
                        satisfied_transition.append(trans)
                if len(satisfied_transition) > 1:
                    raise Exception("ProbabilisticEnv.get_transitions: Bad vehicle state transition configuration")
                if len(satisfied_transition) == 1:
                    trans = satisfied_transition[0]
                    # transition taken
                    for dst, transition_prob in zip(trans.dst, trans.probability):
                        sp.cz_state[cz_id].vehicle_state = dst
                        explore_cz(i + 1, prob * transition_prob, sp)
                        sp.cz_state[cz_id].vehicle_state = trans.src
                    # transition not taken
                    if not math.isclose(0.0, 1.0 - sum(trans.probability)):
                        explore_cz(i + 1, prob * (1.0 - sum(trans.probability)), sp)
                else:
                    # no transition is available
                    explore_cz(i + 1, prob, sp)
            else:
                # no vehicle on this cz
                explore_cz(i + 1, prob, sp)

        def explore_src(i: int, prob: float, sp: PositionBasedStateEnv.DecodedState):
            '''
            Explore the possibility that a vehicle in the queue of a specific source lane starts waiting
            '''
            if i == len(self.sorted_src_lane_ids):
                explore_cz(0, prob, sp)
                return

            src_lane_id = self.sorted_src_lane_ids[i]
            veh_state = sp.src_lane_state[src_lane_id].vehicle_state
            next_pos = sp.src_lane_state[src_lane_id].next_position
            if veh_state == "blocked" and sp.cz_state[next_pos].vehicle_state == "":
                start_waiting_prob = 0.8
                # start waiting
                sp.src_lane_state[src_lane_id].vehicle_state = "waiting"
                explore_src(i + 1, prob * start_waiting_prob, sp)
                sp.src_lane_state[src_lane_id].vehicle_state = "blocked"

                # keep blocked
                explore_src(i + 1, prob * (1 - start_waiting_prob), sp)
            else:
                explore_src(i + 1, prob, sp)

        def explore_queue_dec(i: int, prob: float, sp: PositionBasedStateEnv.DecodedState):
            if i == len(self.sorted_src_lane_ids):
                explore_src(0, prob, sp)
                return

            src_lane_id = self.sorted_src_lane_ids[i]

            # Phase 1: If the front of this source lane is empty, take a vehicle from the queue if its queue size > 0
            if sp.src_lane_state[src_lane_id].vehicle_state == "" \
                and sp.src_lane_state[src_lane_id].queue_size > 0:
                # change to blocked if a vehicle in queue arrived
                transitions = self.transitions_of_src_lane[src_lane_id]
                for next_cz in transitions:
                    sp.src_lane_state[src_lane_id].next_position = next_cz
                    sp.src_lane_state[src_lane_id].vehicle_state = "blocked"

                    # calculate the probability that the queue size decreases for one level
                    cur_level_queue_size = sp.src_lane_state[src_lane_id].queue_size
                    higher_level_queue_size = self.get_higher_level_queue_size(cur_level_queue_size)
                    lower_level_queue_size = self.get_lower_level_queue_size(cur_level_queue_size)
                    diff = higher_level_queue_size - cur_level_queue_size

                    if diff == 0:  # if the current queue size is of the highest level
                        diff = 2
                    
                    assert(diff > 0)  # since we have checked queue size > 0
                    decrease_prob = 1.0 / diff

                    # Case 1: queue size does not decreases
                    if not math.isclose(0.0, 1 - decrease_prob):
                        explore_queue_dec(i + 1, prob * (1 - decrease_prob) * (1 / len(transitions)), sp)
                    
                    # Case 2: queue size decreases
                    if not math.isclose(0.0, decrease_prob):
                        sp.src_lane_state[src_lane_id].queue_size = lower_level_queue_size
                        explore_queue_dec(i + 1, prob * decrease_prob * (1 / len(transitions)), sp) ##
                        sp.src_lane_state[src_lane_id].queue_size = cur_level_queue_size

                    sp.src_lane_state[src_lane_id].next_position = ""
                    sp.src_lane_state[src_lane_id].vehicle_state = ""
            else:
                explore_queue_dec(i + 1, prob, sp)

        def explore_queue_inc(i: int, prob: float, sp: PositionBasedStateEnv.DecodedState):
            '''
            Explore the possibility that a specific source lane's queue size increases for one level
            '''
            if i == len(self.sorted_src_lane_ids):
                explore_queue_dec(0, prob, sp)
                return

            src_lane_id = self.sorted_src_lane_ids[i]
            cur_level_queue_size = origin_s_dec.src_lane_state[src_lane_id].queue_size
            higher_level_queue_size = self.get_higher_level_queue_size(cur_level_queue_size)
            diff = higher_level_queue_size - cur_level_queue_size
            if diff > 0:
                increase_prob = self.traffic_density * (1.0 / diff)
                unchange_prob = 1.0 - increase_prob

                # queue size not increases
                if not math.isclose(0.0, unchange_prob):
                    explore_queue_inc(i + 1, prob * unchange_prob, sp)

                # queue size increases
                if not math.isclose(0.0, increase_prob):
                    sp.src_lane_state[src_lane_id].queue_size = higher_level_queue_size
                    explore_queue_inc(i + 1, prob * increase_prob, sp)
                    sp.src_lane_state[src_lane_id].queue_size = cur_level_queue_size
            else:
                explore_queue_inc(i + 1, prob, sp)

        sp_init = copy.deepcopy(origin_s_dec)
        next_pos: str = ""
        veh_state: str = ""
        state_ref = None
        if a_dec.type == "src":
            state_ref = sp_init.src_lane_state[a_dec.id]
        elif a_dec.type == "cz":
            state_ref = sp_init.cz_state[a_dec.id]
            
        if state_ref is not None:
            next_pos = state_ref.next_position
            veh_state = state_ref.vehicle_state

        if next_pos == "" or veh_state != "waiting":
            explore_queue_inc(0, 1.0, sp_init)
        else:
            state_ref.vehicle_state = ""
            state_ref.next_position = ""

            cost -= 1
            # If there exist other waiting vehicles, then the cost is 0
            # This is for making moving multiple vehicles in one time step possible
            for src_lane_state in sp_init.src_lane_state.values():
                if src_lane_state.vehicle_state == "waiting":
                    cost = 0
            for cz_state in sp_init.cz_state.values():
                if cz_state.vehicle_state == "waiting":
                    cost = 0
            cost /= num_vehicles

            if next_pos == "$":
                explore_queue_inc(0, 1.0, sp_init)
            else:
                trans = self.transitions_of_cz[next_pos]
                for next2_pos in trans:
                    sp_init.cz_state[next_pos] = PositionBasedStateEnv.CzState(
                        vehicle_state="moving",
                        next_position=next2_pos
                    )
                    for src_lane_state in sp_init.src_lane_state.values():
                        if src_lane_state.vehicle_state == "waiting" and src_lane_state.next_position == next_pos:
                            src_lane_state.vehicle_state = "blocked"

                    for cz_state in sp_init.cz_state.values():
                        if cz_state.vehicle_state == "waiting" and cz_state.next_position == next_pos:
                            cz_state.vehicle_state = "blocked"

                    explore_queue_inc(0, 1.0 / len(trans), sp_init)

        self.P[s][a] = res
        return res

    def reachability_analysis(self) -> None:
        state_transition_graph = Digraph()
        print("Conducting reachability analysis...")
        for s in tqdm(range(self.state_space_size), desc="Building state transition graph", ascii=True, leave=False):
            for _, next_s, _ in self.get_transitions(s, 0):
                state_transition_graph.add_edge(s, next_s)
        
        print("Building condensation state transition graph...")
        condensation_state_transition_graph = state_transition_graph.get_scc_graph()

        pbar = tqdm(desc="Calculating reachable states", total=self.state_space_size, leave=False, ascii=True)
        def dfs(scc: Tuple) -> None:
            if self.reachable_states_without_op[scc[0]] is not None:
                return
            self.reachable_states_without_op[scc[0]] = set(scc)
            for child in condensation_state_transition_graph.get_neighbors(scc):
                dfs(child)
                self.reachable_states_without_op[scc[0]].update(self.reachable_states_without_op[child[0]])
            for member in scc[1:]:
                self.reachable_states_without_op[member] = self.reachable_states_without_op[scc[0]]
            pbar.update(len(scc))

        for scc in condensation_state_transition_graph.vertices:
            dfs(scc)
        
        pbar.close()

    def reachable(self, src_state: int, action: int, dst_state: int) -> bool:
        if not self.enable_reachability_analysis:
            raise Exception("ProbabilisticEnv.reachable: called without reachability analysis enabled")
        for _, next_s, _ in self.get_transitions(src_state, action):
            if dst_state in self.reachable_states_without_op[next_s]:
                return True
        return False
