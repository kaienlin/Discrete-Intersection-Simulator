from __future__ import annotations

import gym
from typing import Literal, Tuple, Dict, List
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

        self.state_to_int: Dict[Tuple[VehicleBasedStateEnv.VehicleState], int] = {}
        self.int_to_state: List[Tuple[VehicleBasedStateEnv.VehicleState]] = []

    @property
    def state_space_size(self) -> int:
        return len(self.int_to_state)

    @property
    def action_space_size(self) -> int:
        return self.max_vehicle_num + 1

    def is_actable_state(self, s: int) -> int:
        vehicles = self.decode_state(s)
        return any([v.state == "waiting" for v in vehicles])

    def is_effective_action_of_state(self, a: int, s: int) -> bool:
        if a == 0:
            return True
        vehicles = self.decode_state(s)
        return not (a > len(vehicles) or vehicles[a - 1].state != "waiting")

    @lru_cache(maxsize=2)
    def encode_state(self, state: Tuple[VehicleBasedStateEnv.VehicleState]) -> int:
        state = tuple(sorted(state))
        res = self.state_to_int.get(state, len(self.state_to_int))
        if res < len(self.state_to_int):
            return res
        self.state_to_int[state] = res
        self.int_to_state.append(state)
        return res

    def decode_state(self, s: int) -> VehicleBasedStateEnv.VehicleState:
        if s < len(self.int_to_state):
            return self.int_to_state[s]
        raise Exception("VehicleBasedStateEnv: trying to decode unseen state")
    
    @dataclass(order=True, unsafe_hash=True)
    class VehicleState:
        src_lane: str = ""
        trajectory: Tuple[str] = field(default_factory=tuple)
        position: int = -1
        state: Literal["waiting", "blocked", "moving"] = "moving"
