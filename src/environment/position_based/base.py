from __future__ import annotations

from typing import Tuple, Dict, Set, List
from functools import lru_cache
from dataclasses import dataclass, field

from tqdm import tqdm
from gym import spaces
import gym

from simulation.intersection import Intersection


class PositionBasedStateEnv(gym.Env):
    TERMINAL_STATE = 0
    DEADLOCK_COST = 1e9
    vehicle_states_in_cz: Tuple[str] = ("waiting", "blocked", "moving")
    vehicle_states_in_src: Tuple[str] = ("waiting", "blocked")

    def __init__(
        self,
        intersection: Intersection,
        queue_size_scale: Tuple[int] = (1,),
    ):
        super().__init__()
        self.intersection: Intersection = intersection
        self.queue_size_scale: Tuple[int] = queue_size_scale

        if len(queue_size_scale) == 0 or queue_size_scale[0] != 1:
            raise Exception("BaseIntersectionEnv: Invalid queue size scale")

        self.sorted_src_lane_ids: List[str] = sorted(intersection.src_lanes.keys())
        self.sorted_cz_ids: List[str] = sorted(intersection.conflict_zones)

        self.state_space_size: int = self._create_state_encoding()
        self.action_space_size: int = self._create_action_encoding()

        self.observation_space = spaces.Discrete(self.state_space_size)
        self.action_space = spaces.Discrete(self.action_space_size)

        self.deadlock_state_table = [False for _ in range(self.state_space_size)]
        for s in tqdm(range(self.state_space_size), desc="Finding deadlock states", leave=False, ascii=True):
            raw_s = self.compressed_to_raw_state[s]
            if self._is_deadlock_raw_state(raw_s):
                self.deadlock_state_table[s] = True

    def _create_state_encoding(self) -> int:
        # find all valid transitions
        transitions_of_cz: Dict[str, Set[str]] = {cz_id: set() for cz_id in self.intersection.conflict_zones}
        for traj in sorted(self.intersection.trajectories):
            for idx, cz_1 in enumerate(traj[:-1]):
                cz_2 = traj[idx + 1]
                transitions_of_cz[cz_1].add(cz_2)
            transitions_of_cz[traj[-1]].add("$")
        self.transitions_of_cz: Dict[str, List[str]] = {k: sorted(s) for k, s in transitions_of_cz.items()}
        self.transitions_of_src_lane: Dict[str, List[str]] = {k: sorted(s) for k, s in self.intersection.src_lanes.items()}

        n_raw_states: int = 1

        # fields encoding vehicles' positions and states
        self.cz_field_width: Dict[str, int] = {}
        self.src_lane_field_width: Dict[str, int] = {}

        for src_lane_id, associated_czs in sorted(self.intersection.src_lanes.items()):
            field_width = len(self.vehicle_states_in_src) * len(associated_czs) + 1
            n_raw_states *= field_width
            self.src_lane_field_width[src_lane_id] = field_width

        for cz_id, trans in sorted(transitions_of_cz.items()):
            field_width = len(self.vehicle_states_in_cz) * len(trans) + 1
            n_raw_states *= field_width
            self.cz_field_width[cz_id] = field_width

        # fields encoding the queue sizes of the source lanes
        n_raw_states *= (len(self.queue_size_scale) + 1) ** len(self.sorted_src_lane_ids)

        # compress state space by filtering out invalid states
        n_compressed_states: int = 0
        compressed_to_raw_state: Dict[int, int] = dict()
        raw_to_compressed_state: Dict[int, int] = dict()
        for raw_s in tqdm(range(n_raw_states), desc="Filtering out invalid states", leave=False, ascii=True):
            if not self._is_invalid_raw_state(raw_s):
                compressed_to_raw_state[n_compressed_states] = raw_s
                raw_to_compressed_state[raw_s] = n_compressed_states
                n_compressed_states += 1
        self.compressed_to_raw_state: Dict[int, int] = compressed_to_raw_state
        self.raw_to_compressed_state: Dict[int, int] = raw_to_compressed_state

        return n_compressed_states

    def _create_action_encoding(self) -> int:
        return len(self.intersection.conflict_zones) \
               + len(self.intersection.src_lanes) + 1

    def _is_invalid_raw_state(self, raw_state: int) -> bool:
        decoded_state: PositionBasedStateEnv.DecodedState = self._decode_raw_state(raw_state)
        occupied_cz: Set = set()
        for cz_id, cz_state in decoded_state.cz_state.items():
            if cz_state.next_position:
                occupied_cz.add(cz_id)
        for _, src_lane_state in decoded_state.src_lane_state.items():
            if src_lane_state.vehicle_state == "waiting" \
                and src_lane_state.next_position in occupied_cz:
                return True
            if src_lane_state.vehicle_state == "" and src_lane_state.queue_size > 0:
                return True
        for _, cz_state in decoded_state.cz_state.items():
            if cz_state.vehicle_state == "waiting" \
                and cz_state.next_position in occupied_cz:
                return True
        return False

    def is_invalid_state(self, state: int) -> bool:
        raw_state = self.compressed_to_raw_state[state]
        return self._is_invalid_raw_state(raw_state)

    def _is_deadlock_raw_state(self, raw_state: int) -> bool:
        decoded_state: PositionBasedStateEnv.DecodedState = self._decode_raw_state(raw_state)
        adj = {cz_id: [] for cz_id in self.sorted_cz_ids}
        for cz_id, cz_state in decoded_state.cz_state.items():
            next_pos = cz_state.next_position
            if next_pos and next_pos != "$" and cz_state.vehicle_state == "blocked":
                adj[cz_id].append(next_pos)

        color = {cz_id: "w" for cz_id in self.sorted_cz_ids}
        def dfs(v: str):
            color[v] = "g"
            for u in adj[v]:
                if color[u] == "w":
                    if dfs(u):
                        return True
                elif color[u] == "g":
                    return True
            color[v] = "b"
            return False

        for cz_id in self.sorted_cz_ids:
            if color[cz_id] == "w":
                if dfs(cz_id):
                    return True

        return False

    def is_deadlock_state(self, state: int) -> bool:
        return self.deadlock_state_table[state]

    def _is_actable_raw_state(self, raw_state: int) -> bool:
        decoded_state: PositionBasedStateEnv.DecodedState = self._decode_raw_state(raw_state)
        for src_lane_state in decoded_state.src_lane_state.values():
            if src_lane_state.vehicle_state == "waiting":
                return True
        for cz_state in decoded_state.cz_state.values():
            if cz_state.vehicle_state == "waiting":
                return True
        return False
    
    def is_actable_state(self, state: int) -> bool:
        raw_state = self.compressed_to_raw_state[state]
        return self._is_actable_raw_state(raw_state)

    def is_effective_action_of_state(self, action: int, state: int) -> bool:
        decoded_state: PositionBasedStateEnv.DecodedState = self.decode_state(state)
        decoded_action: PositionBasedStateEnv.DecodedAction = self.decode_action(action)
        if decoded_action.type == "":
            return True
        if decoded_action.type == "src" and decoded_state.src_lane_state[decoded_action.id].vehicle_state == "waiting":
            return True
        if decoded_action.type == "cz" and decoded_state.cz_state[decoded_action.id].vehicle_state == "waiting":
            return True
        return False

    def _discretize_queue_size(self, queue_size: int) -> int:
        for idx, threshold in enumerate(self.queue_size_scale):
            if queue_size < threshold:
                return idx
        return len(self.queue_size_scale)

    def _undiscretize_queue_size(self, discretized_queue_size: int) -> int:
        if discretized_queue_size <= 0:
            return 0
        if discretized_queue_size > len(self.queue_size_scale):
            discretized_queue_size = len(self.queue_size_scale)
        return self.queue_size_scale[discretized_queue_size - 1]

    def encode_state(self, decoded_state: DecodedState) -> int:
        state = 0
        for cz_id in self.sorted_cz_ids:
            state *= self.cz_field_width[cz_id]
            trans = self.transitions_of_cz[cz_id]
            next_pos = decoded_state.cz_state[cz_id].next_position
            veh_state = decoded_state.cz_state[cz_id].vehicle_state
            if next_pos:
                state += trans.index(next_pos) * len(self.vehicle_states_in_cz)
                state += self.vehicle_states_in_cz.index(veh_state)
                state += 1

        for src_lane_id in self.sorted_src_lane_ids:
            state *= self.src_lane_field_width[src_lane_id]
            trans = self.transitions_of_src_lane[src_lane_id]
            veh_state = decoded_state.src_lane_state[src_lane_id].vehicle_state
            next_pos = decoded_state.src_lane_state[src_lane_id].next_position
            if next_pos:
                state += trans.index(next_pos) * len(self.vehicle_states_in_src)
                state += self.vehicle_states_in_src.index(veh_state)
                state += 1

        for src_lane_id in self.sorted_src_lane_ids:
            state *= len(self.queue_size_scale) + 1
            queue_size = decoded_state.src_lane_state[src_lane_id].queue_size
            discretized_queue_size = self._discretize_queue_size(queue_size)
            state += discretized_queue_size

        return self.raw_to_compressed_state[state]

    def make_decoded_state(self) -> DecodedState:
        res = self.DecodedState()
        res.src_lane_state = {src_lane_id: self.SrcLaneState() for src_lane_id in self.sorted_src_lane_ids}
        res.cz_state = {cz_id: self.CzState() for cz_id in self.sorted_cz_ids}
        return res

    @lru_cache(maxsize=1024)
    def _decode_raw_state(self, raw_state: int) -> DecodedState:
        res = self.make_decoded_state()
        for src_lane_id in self.sorted_src_lane_ids[::-1]:
            discretized_queue_size = raw_state % (len(self.queue_size_scale) + 1)
            raw_state //= (len(self.queue_size_scale) + 1)
            queue_size = self._undiscretize_queue_size(discretized_queue_size)
            res.src_lane_state[src_lane_id].queue_size = queue_size

        for src_lane_id in self.sorted_src_lane_ids[::-1]:
            trans = self.transitions_of_src_lane[src_lane_id]
            lane_state = raw_state % self.src_lane_field_width[src_lane_id]
            raw_state //= self.src_lane_field_width[src_lane_id]
            if lane_state > 0:
                lane_state -= 1
                veh_state = self.vehicle_states_in_src[lane_state % len(self.vehicle_states_in_src)]
                lane_state //= len(self.vehicle_states_in_src)
                res.src_lane_state[src_lane_id].vehicle_state = veh_state
                res.src_lane_state[src_lane_id].next_position = trans[lane_state]

        for cz_id in self.sorted_cz_ids[::-1]:
            trans = self.transitions_of_cz[cz_id]
            field_width = self.cz_field_width[cz_id]
            cz_state = raw_state % field_width
            raw_state //= field_width
            if cz_state > 0:
                cz_state -= 1
                veh_state = self.vehicle_states_in_cz[cz_state % len(self.vehicle_states_in_cz)]
                cz_state //= len(self.vehicle_states_in_cz)
                res.cz_state[cz_id].vehicle_state = veh_state
                res.cz_state[cz_id].next_position = trans[cz_state]
        
        assert(raw_state == 0)
        return res

    def decode_state(self, state: int) -> DecodedState:
        raw_state = self.compressed_to_raw_state[state]
        return self._decode_raw_state(raw_state)

    def encode_action(self, decoded_action: DecodedAction) -> int:
        if decoded_action.type == "":
            return 0
        num_src_lane = len(self.sorted_src_lane_ids)
        if decoded_action.type == "src":
            return 1 + self.sorted_src_lane_ids.index(decoded_action.id)
        if decoded_action.type == "cz":
            return 1 + num_src_lane + self.sorted_cz_ids.index(decoded_action.id)

    @lru_cache(maxsize=16)
    def decode_action(self, action: int) -> DecodedAction:
        if action == 0:
            return PositionBasedStateEnv.DecodedAction()
        num_src_lane = len(self.sorted_src_lane_ids)
        num_cz = len(self.sorted_src_lane_ids)
        if 1 <= action <= num_src_lane:
            return PositionBasedStateEnv.DecodedAction(type="src", id=self.sorted_src_lane_ids[action - 1])
        elif num_src_lane + 1 <= action <= num_cz + num_src_lane:
            return PositionBasedStateEnv.DecodedAction(type="cz", id=self.sorted_cz_ids[action - num_src_lane - 1])
        else:
            raise Exception(f"BaseIntersectionEnv.decode_action: Invalid action {action}")


    @dataclass
    class SrcLaneState:
        vehicle_state: str = ""
        next_position: str = ""
        queue_size: int = 0

    @dataclass
    class CzState:
        vehicle_state: str = ""
        next_position: str = ""

    @dataclass
    class DecodedState:
        src_lane_state: Dict[str, PositionBasedStateEnv.SrcLaneState] = field(default_factory=dict)
        cz_state: Dict[str, PositionBasedStateEnv.CzState] = field(default_factory=dict)

        def print(self) -> None:
            print("1. Source lane states:")
            for src_lane_id, src_lane_state in self.src_lane_state.items():
                print(f"  {src_lane_id}: ", end="")
                print(f"queue_size={src_lane_state.queue_size: <7} ", end="")
                if src_lane_state.next_position:
                    print(f"vehicle_state={src_lane_state.vehicle_state: <7}, ", end="")
                    print(f"next_position={src_lane_state.next_position: <7}", end="")
                print("")

            print("2. Conflict zone states:")
            for cz_id, cz_state in self.cz_state.items():
                if cz_state.next_position:
                    print(f"  {cz_id}: ", end="")
                    print(f"vehicle_state={cz_state.vehicle_state: <7}, ", end="")
                    print(f"next_position={cz_state.next_position: <7}", end="")
                    print("")

    @dataclass
    class DecodedAction:
        type: str = ""
        id: str = ""

        def print(self) -> None:
            print(f"Action: type={self.type: <7}, id={self.id: <7}")
