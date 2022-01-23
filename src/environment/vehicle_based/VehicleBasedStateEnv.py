from __future__ import annotations

import gym
from typing import Literal, Tuple, Dict, Set, List
from functools import lru_cache
from dataclasses import dataclass, field

from simulator.intersection import Intersection

class VehicleBasedStateEnv(gym.Env):
    DEADLOCK_COST = 1e9
    vehicle_state_values: Tuple[str] = ("waiting", "blocked", "moving")

    def __init__(
        self,
        intersection: Intersection,
        max_vehicle_num: int,
    ):
        super().__init__()
        self.intersection: Intersection = intersection
        self.max_vehicle_num: int = max_vehicle_num

        self.state_to_int: Dict[VehicleBasedStateEnv.VehicleState, int] = {}
        self.int_to_state: List[VehicleBasedStateEnv.VehicleState] = []

    @property
    def state_space_size(self) -> int:
        return len(self.int_to_state)

    @property
    def action_space_size(self) -> int:
        return self.max_vehicle_num

    def encode_state(self, state: VehicleBasedStateEnv.VehicleState) -> int:
        res = self.state_to_int.get(state, len(self.state_to_int))
        if res < len(self.state_to_int):
            return res
        self.state_to_int[state] = res
        self.int_to_state.append(state)
        return res

    def decode_state(self, s: int) -> VehicleBasedStateEnv.VehicleState:
        if s < len(self.state_dict):
            return self.int_to_state[s]
        raise Exception("VehicleBasedStateEnv: trying to decode unseen state")
    
    @dataclass(order=True, unsafe_hash=True)
    class VehicleState:
        src_lane: str = ""
        trajectory: Tuple[str] = field(default_factory=tuple)
        position: int = -1
        state: Literal["waiting", "blocked", "moving"] = "moving"
