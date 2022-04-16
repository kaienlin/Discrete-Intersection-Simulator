from __future__ import annotations

import enum
from typing import Dict, Tuple, Set

class VehicleState(enum.Enum):
    NOT_ARRIVED = enum.auto()
    READY = enum.auto()
    MOVING = enum.auto()
    BLOCKED = enum.auto()
    LEFT = enum.auto()


class Vehicle:
    def __init__(
        self,
        _id: str,
        earliest_arrival_time: int,
        trajectory: Tuple[str],
        src_lane_id: str,
        dst_lane_id: str,
        vertex_passing_time: int = 10
    ) -> None:
        self.__id: str = _id
        self.__earliest_arrival_time: int = earliest_arrival_time
        self.__trajectory: Tuple[str] = trajectory
        self.__src_lane_id: str = src_lane_id
        self.__dst_lane_id: str = dst_lane_id
        self.__vertex_passing_time: int = vertex_passing_time

        # simulation-related attributes
        self.__state: VehicleState = VehicleState.NOT_ARRIVED
        self.__idx_on_traj: int = -1

    def reset(self) -> None:
        self.__state = VehicleState.NOT_ARRIVED
        self.__idx_on_traj = -1

    def on_last_cz(self) -> bool:
        return self.__idx_on_traj == len(self.trajectory) - 1

    def move_to_next_cz(self) -> None:
        if self.__state != VehicleState.READY:
            raise Exception("[Vehicle.move_to_next_CZ] invalid vehicle state")
        self.__idx_on_traj += 1

    def set_state(self, state: VehicleState) -> None:
        self.__state = state

    def get_cur_cz(self) -> str:
        if self.idx_on_traj == -1:
            return "^"
        if self.idx_on_traj == len(self.__trajectory):
            return "$"
        return self.__trajectory[self.__idx_on_traj]
    
    def get_next_cz(self) -> str:
        if self.__idx_on_traj == len(self.__trajectory) - 1:
            return "$"
        if self.__idx_on_traj == len(self.__trajectory):
            return " "
        return self.__trajectory[self.__idx_on_traj + 1]

    def asdict(self) -> Dict:
        res = {}
        res["id"] = self.__id
        res["earliest_arrival_time"] = self.__earliest_arrival_time
        res["trajectory"] = self.__trajectory
        res["src_lane_id"] = self.__src_lane_id
        res["dst_lane_id"] = self.__dst_lane_id
        res["vertex_passing_time"] = self.__vertex_passing_time
        return res

    def __hash__(self):
        return hash(self.__id)

    def __eq__(self, other: object):
        if not isinstance(other, Vehicle):
            raise NotImplementedError()
        return self.__id == other.id

    @property
    def id(self) -> str:
        return self.__id

    @property
    def earliest_arrival_time(self) -> int:
        return self.__earliest_arrival_time

    @property
    def trajectory(self) -> Tuple[str]:
        return self.__trajectory

    @property
    def src_lane_id(self) -> str:
        return self.__src_lane_id

    @property
    def dst_lane_id(self) -> str:
        return self.__dst_lane_id

    @property
    def vertex_passing_time(self) -> int:
        return self.__vertex_passing_time

    @property
    def state(self) -> VehicleState:
        return self.__state

    @property
    def idx_on_traj(self) -> int:
        return self.__idx_on_traj

