from environment.tabular.vehicle_based import VehicleBasedStateEnv
from utility import Digraph

from .base import Policy


class SingleCzPolicy(Policy):
    def __init__(self, env):
        self.env = env
        assert isinstance(env, VehicleBasedStateEnv)

    def decide(self, state: int) -> int:
        decoded_state = self.env.decode_state(state)
        G = Digraph()
        for vehicle_state in decoded_state:
            if 0 <= vehicle_state.position <= len(vehicle_state.trajectory) - 2:
                cur_cz = vehicle_state.trajectory[vehicle_state.position]
                next_cz = vehicle_state.trajectory[vehicle_state.position + 1]
                G.add_edge(cur_cz, next_cz)
        
        available = True
        for i, vehicle_state in enumerate(decoded_state):
            if vehicle_state.position == 0:
                available = False
            if vehicle_state.position == 0 and vehicle_state.state == "waiting":
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
        
        if not available:
            return 0
        
        for i, vehicle_state in enumerate(decoded_state):
            if vehicle_state.state == "waiting":
                return i + 1
        
        return 0


class RGS(Policy):
    def __init__(self, env, tau: int = 2):
        self.env = env
        self.tau = tau
        assert isinstance(env, VehicleBasedStateEnv)

        self.dependency_cycles = [
            {("1", "3"), ("3", "4"), ("4", "2"), ("2", "1")}
        ]
        if len(env.sim.intersection.conflict_zones) == 2:
            self.dependency_cycles = [
                {("1", "2"), ("2", "1")}
            ]

    def decide(self, state: int) -> int:
        decoded_state = self.env.decode_state(state)

        transitions = set()
        for vehicle_state in decoded_state:
            if vehicle_state.position >= 0 and vehicle_state.position < len(vehicle_state.trajectory) - 1:
                transitions.add(
                    (vehicle_state.trajectory[vehicle_state.position], vehicle_state.trajectory[vehicle_state.position + 1])
                )

        ok = True
        for C in self.dependency_cycles:
            if len(transitions.intersection(C)) > len(C) - self.tau:
                ok = False
                break
        
        if not ok:
            for i, vehicle_state in enumerate(decoded_state):
                if vehicle_state.state == "waiting" and vehicle_state.position >= 0:
                    if vehicle_state.position == len(vehicle_state.trajectory) - 1:
                        return i + 1
                    cur_cz = vehicle_state.trajectory[vehicle_state.position]
                    next_cz = vehicle_state.trajectory[vehicle_state.position + 1]
                    if any((cur_cz, next_cz) in C for C in self.dependency_cycles):
                        return i + 1
            return 0

        for i, vehicle_state in enumerate(decoded_state):
            if vehicle_state.state == "waiting":
                if vehicle_state.position >= len(vehicle_state.trajectory) - 2:
                    return i + 1
                next_cz = vehicle_state.trajectory[vehicle_state.position + 1]
                next2_cz = vehicle_state.trajectory[vehicle_state.position + 2]

                if vehicle_state.position >= 0:
                    cur_cz = vehicle_state.trajectory[vehicle_state.position]
                    transitions.remove((cur_cz, next_cz))

                transitions.add((next_cz, next2_cz))
                ok = True
                for C in self.dependency_cycles:
                    if len(transitions.intersection(C)) > len(C) - self.tau:
                        ok = False
                if ok:
                    return i + 1
                
                transitions.remove((next_cz, next2_cz))
                if vehicle_state.position >= 0:
                    cur_cz = vehicle_state.trajectory[vehicle_state.position]
                    transitions.add((cur_cz, next_cz))

        return 0
