from typing import Dict, Set
import itertools

from .base import Policy
from environment.PositionBasedStateEnv import PositionBasedStateEnv
from utility import Digraph

class IGreedyPolicy(Policy):
    def __init__(self, env: PositionBasedStateEnv):
        self.env = env
        self.transitions_of_cz: Dict[str, Set[str]] = {cz_id: set() for cz_id in env.intersection.conflict_zones}

        for src, dst in env.intersection.transitions:
            self.transitions_of_cz[src].add(dst)

    def decide(self, state: int) -> int:
        decoded_state = self.env.decode_state(state)
        G = Digraph()

        for cz_id, cz_state in decoded_state.cz_state.items():
            if cz_state.next_position not in ["", "$"]:
                G.add_edge(cz_id, cz_state.next_position)
        
        positions = itertools.chain(decoded_state.cz_state.items(), decoded_state.src_lane_state.items())
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
                    t: str = "src" if isinstance(pos_state, PositionBasedStateEnv.SrcLaneState) else "cz"
                    action = PositionBasedStateEnv.DecodedAction(type=t, id=pos)
                    return self.env.encode_action(action)

        return self.env.encode_action(PositionBasedStateEnv.DecodedAction(type="", id=""))
