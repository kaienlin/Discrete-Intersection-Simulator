from __future__ import annotations

import gym
from gym import spaces
from typing import Tuple, Dict, Set, List
from functools import lru_cache
import copy, math, sys
from dataclasses import dataclass, field

from simulator.intersection import Intersection

class BaseIntersectionEnv(gym.Env):
    TERMINAL_STATE = 0
    DEADLOCK_COST = 1e9

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
        src_lane_state: Dict[str, BaseIntersectionEnv.SrcLaneState] = field(default_factory=dict)
        cz_state: Dict[str, BaseIntersectionEnv.CzState] = field(default_factory=dict)

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
    
    def __init__(
        self,
        intersection: Intersection,
        queue_size_scale: Tuple[int] = (1,),
        vehicle_states_in_cz: Tuple[str] = ("waiting", "blocked", "moving"),
        traffic_density: float = 0.05
    ):
        super(BaseIntersectionEnv, self).__init__()
        self.intersection: Intersection = intersection
        self.queue_size_scale: Tuple[int] = queue_size_scale
        self.vehicle_states_in_cz: Tuple[str] = vehicle_states_in_cz
        self.traffic_density: float = traffic_density

        self.sorted_src_lane_ids: List[str] = sorted(intersection.src_lanes.keys())
        self.sorted_cz_ids: List[str] = sorted(intersection.conflict_zones)

        self.state_space_size: int = self._create_state_encoding()
        self.action_space_size: int = self._create_action_encoding()

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
            field_width = len(associated_czs) + 1
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
        for raw_s in range(n_raw_states):
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
        decoded_state: BaseIntersectionEnv.DecodedState = self._decode_raw_state(raw_state)
        occupied_cz: Set = set()
        for cz_id, cz_state in decoded_state.cz_state.items():
            if cz_state.next_position:
                occupied_cz.add(cz_id)
        for _, src_lane_state in decoded_state.src_lane_state.items():
            if src_lane_state.vehicle_state == "waiting" \
                and src_lane_state.next_position in occupied_cz:
                return True
        for _, cz_state in decoded_state.cz_state.items():
            if cz_state.vehicle_state == "waiting" \
                and cz_state.next_position in occupied_cz:
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
            next_pos = decoded_state.src_lane_state[src_lane_id].next_position
            if next_pos:
                state += trans.index(next_pos) + 1
        
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
                res.src_lane_state[src_lane_id].vehicle_state = "waiting"
                res.src_lane_state[src_lane_id].next_position = trans[lane_state - 1]
            
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

