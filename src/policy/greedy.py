from typing import Dict, Set, Union, Tuple, Iterable
import itertools

from utility import Digraph
from simulation import Vehicle, VehicleState
from environment import RawStateSimulatorEnv
from environment.tabular.position_based import PositionBasedStateEnv
from environment.tabular.vehicle_based import VehicleBasedStateEnv

from .base import Policy


class IGreedyPolicy(Policy):
    def __init__(self, env: Union[PositionBasedStateEnv, VehicleBasedStateEnv]):
        self.env: Union[PositionBasedStateEnv, VehicleBasedStateEnv] = env
        self.transitions_of_cz: Dict[str, Set[str]] = {
            cz_id: set() for cz_id in env.intersection.conflict_zones}

        for src, dst in env.intersection.transitions:
            self.transitions_of_cz[src].add(dst)

    def decide(self, state: int) -> int:
        if isinstance(self.env, PositionBasedStateEnv):
            return self.__decide_position_based(state)
        if isinstance(self.env, VehicleBasedStateEnv):
            return self.__decide_vehicle_based(state)
        if isinstance(self.env, RawStateSimulatorEnv):
            return self.__decide_raw(state)
        assert False

    def __decide_position_based(self, state: int) -> int:
        decoded_state = self.env.decode_state(state)
        G = Digraph()

        for cz_id, cz_state in decoded_state.cz_state.items():
            if cz_state.next_position not in ["", "$"]:
                G.add_edge(cz_id, cz_state.next_position)

        positions = itertools.chain(
            decoded_state.cz_state.items(), decoded_state.src_lane_state.items())
        for pos, pos_state in positions:
            if pos_state.vehicle_state == "waiting":
                safe = True
                if pos_state.next_position != "$":
                    for next2pos in self.transitions_of_cz[pos_state.next_position]:
                        G.add_edge(pos_state.next_position, next2pos)
                        if isinstance(pos_state, PositionBasedStateEnv.CzState):
                            G.remove_edge(pos, pos_state.next_position)
                        cyclic = G.has_cycle()
                        G.remove_edge(pos_state.next_position, next2pos)
                        if isinstance(pos_state, PositionBasedStateEnv.CzState):
                            G.add_edge(pos, pos_state.next_position)
                        if cyclic:
                            safe = False
                            break
                if safe:
                    t: str = "src" \
                        if isinstance(pos_state, PositionBasedStateEnv.SrcLaneState) \
                        else "cz"
                    action = PositionBasedStateEnv.DecodedAction(type=t, id=pos)
                    return self.env.encode_action(action)

        return self.env.encode_action(PositionBasedStateEnv.DecodedAction(type="", id=""))

    def __decide_vehicle_based(self, state: int) -> int:
        decoded_state: Tuple[VehicleBasedStateEnv.VehicleState] = self.env.decode_state(state)
        G = Digraph()
        for vehicle_state in decoded_state:
            if 0 <= vehicle_state.position <= len(vehicle_state.trajectory) - 2:
                cur_cz = vehicle_state.trajectory[vehicle_state.position]
                next_cz = vehicle_state.trajectory[vehicle_state.position + 1]
                G.add_edge(cur_cz, next_cz)

        for i, vehicle_state in enumerate(decoded_state):
            if vehicle_state.state == "waiting":
                if vehicle_state.position >= len(vehicle_state.trajectory) - 2:
                    return i + 1
                cz1 = vehicle_state.trajectory[vehicle_state.position + 1]
                cz2 = vehicle_state.trajectory[vehicle_state.position + 2]
                G.add_edge(cz1, cz2)
                if vehicle_state.position > -1:
                    G.remove_edge(vehicle_state.trajectory[vehicle_state.position], cz1)
                cyclic = G.has_cycle()
                G.remove_edge(cz1, cz2)
                if vehicle_state.position > -1:
                    G.add_edge(vehicle_state.trajectory[vehicle_state.position], cz1)
                if not cyclic:
                    return i + 1

        return 0

    def __decide_raw(self, state: Iterable[Vehicle]) -> int:
        G = Digraph()
        for vehicle in state:
            if 0 <= vehicle.idx_on_traj <= len(vehicle.trajectory) - 2:
                cur_cz = vehicle.get_cur_cz()
                next_cz = vehicle.get_next_cz()
                G.add_edge(cur_cz, next_cz)

        for i, vehicle in enumerate(state):
            if vehicle.state == VehicleState.WAITING:
                if vehicle.idx_on_traj >= len(vehicle.trajectory) - 2:
                    return i + 1
                cz1 = vehicle.trajectory[vehicle.idx_on_traj + 1]
                cz2 = vehicle.trajectory[vehicle.idx_on_traj + 2]
                G.add_edge(cz1, cz2)
                if vehicle.idx_on_traj > -1:
                    G.remove_edge(vehicle.get_cur_cz(), cz1)
                cyclic = G.has_cycle()
                G.remove_edge(cz1, cz2)
                if vehicle.idx_on_traj > -1:
                    G.add_edge(vehicle.get_cur_cz(), cz1)
                if not cyclic:
                    return i + 1

        return 0
