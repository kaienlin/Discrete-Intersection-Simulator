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

        self.encoding_table: Dict[Tuple[VehicleBasedStateEnv.VehicleState], int] = {}
        self.decoding_table: List[Tuple[VehicleBasedStateEnv.VehicleState]] = []

    @property
    def state_space_size(self) -> int:
        return len(self.decoding_table)

    @property
    def action_space_size(self) -> int:
        return self.max_vehicle_num + 1

    def is_actable_state(self, state: int) -> int:
        decoded_state = self.decode_state(state)
        return any([v.state == "waiting" for v in decoded_state])

    def is_effective_action_of_state(self, action: int, state: int) -> bool:
        if action == 0:
            return True
        vehicles = self.decode_state(state)
        return not (action > len(vehicles) or vehicles[action - 1].state != "waiting")

    @lru_cache(maxsize=2)
    def encode_state(self, decoded_state: Tuple[VehicleBasedStateEnv.VehicleState]) -> int:
        decoded_state = tuple(sorted(decoded_state))
        res = self.encoding_table.get(decoded_state, len(self.encoding_table))
        if res < len(self.encoding_table):
            return res
        self.encoding_table[decoded_state] = res
        self.decoding_table.append(decoded_state)
        return res

    def decode_state(self, state: int) -> Tuple[VehicleBasedStateEnv.VehicleState]:
        if state < len(self.decoding_table):
            return self.decoding_table[state]
        raise Exception("VehicleBasedStateEnv: trying to decode unseen state")
    
    @dataclass(order=True, unsafe_hash=True)
    class VehicleState:
        src_lane: str = ""
        trajectory: Tuple[str] = field(default_factory=tuple)
        position: int = -1
        state: Literal["waiting", "blocked", "moving"] = "moving"
