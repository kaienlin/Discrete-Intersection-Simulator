import enum
from typing import Tuple, Set

class VehicleState(enum.Enum):
    NOT_ARRIVED = enum.auto()
    WAITING_ON_SRC = enum.auto()
    WAITING_ON_TRAJ = enum.auto()
    MOVING_IN_CZ = enum.auto()
    MOVING_IN_TRANS = enum.auto()
    BLOCKED = enum.auto()
    LEAVED = enum.auto()


class Vehicle:
    def __init__(
        self,
        _id: str,
        earliest_arrival_time: int,
        trajectory: Tuple[str],
        src_lane_id: str,
        dst_lane_id: str = "NOT SET",
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
        self.valid_transitions: Set[Tuple[VehicleState, VehicleState]] = {
            (VehicleState.NOT_ARRIVED, VehicleState.WAITING_ON_SRC),
            (VehicleState.NOT_ARRIVED, VehicleState.BLOCKED),
            (VehicleState.WAITING_ON_SRC, VehicleState.MOVING_IN_CZ),
            (VehicleState.WAITING_ON_TRAJ, VehicleState.MOVING_IN_CZ),
            (VehicleState.MOVING_IN_CZ, VehicleState.WAITING_ON_TRAJ),
            (VehicleState.MOVING_IN_CZ, VehicleState.BLOCKED),
            (VehicleState.BLOCKED, VehicleState.MOVING_IN_TRANS),
            (VehicleState.MOVING_IN_CZ, VehicleState.LEAVED)
        }

    def move_to_next_CZ(self) -> None:
        if self.__state != VehicleState.WAITING_ON_TRAJ:
            raise Exception("[Vehicle.move_to_next_CZ] invalid vehicle state")
        self.__idx_on_traj += 1

    def set_state(self, state: VehicleState) -> None:
        if (self.__state, state) in self.valid_transitions:
            raise Exception("[Vehicle.set_state] invalid state transition")
        self.__state = state

