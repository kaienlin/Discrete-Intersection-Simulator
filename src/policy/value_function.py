import numpy as np

from .base import Policy
from .greedy import IGreedyPolicy
from utility import Digraph

class QTablePolicy(Policy):
    def __init__(self, env, Q):
        self.env = env
        self.Q = Q
        self.original_state_space_size = self.env.state_space_size
        
        self.prev_state: int = -1
        self.prev_action: int = -1
        self.waiting_counter: int = 0
        self.max_waiting = 100

        self.igreedy = IGreedyPolicy(self.env)

    def decide(self, state: int) -> int:
        if state >= self.original_state_space_size:
            return self.igreedy.decide(state)

        Q_state = self.Q[state]
        effective_actions = [a for a in range(self.env.action_space_size) if self.env.is_effective_action_of_state(a, state)]

        G = Digraph()
        decoded_state = self.env.decode_state(state)
        for vehicle_state in decoded_state:
            if 0 <= vehicle_state.position <= len(vehicle_state.trajectory) - 2:
                cur_cz = vehicle_state.trajectory[vehicle_state.position]
                next_cz = vehicle_state.trajectory[vehicle_state.position + 1]
                G.add_edge(cur_cz, next_cz)

        for action in list(effective_actions):
            if action == 0:
                continue
            vehicle_state = decoded_state[action - 1]
            if vehicle_state.position >= len(vehicle_state.trajectory) - 2:
                continue
            cz1 = vehicle_state.trajectory[vehicle_state.position + 1]
            cz2 = vehicle_state.trajectory[vehicle_state.position + 2]
            G.add_edge(cz1, cz2)
            if vehicle_state.position > -1:
                G.remove_edge(vehicle_state.trajectory[vehicle_state.position], cz1)
            cyclic = G.has_cycle()
            G.remove_edge(cz1, cz2)
            if vehicle_state.position > -1:
                G.add_edge(vehicle_state.trajectory[vehicle_state.position], cz1)
            if cyclic:
                effective_actions.remove(action)

        # block NO-OP action
        if 0 in effective_actions and len(effective_actions) > 1 and self.igreedy.decide(state) != 0:
            effective_actions.remove(0)

        action = effective_actions[Q_state[effective_actions].argmin()]

        if action == 0:
            if state == self.prev_state:
                self.waiting_counter += 1
            else:
                self.waiting_counter = 1
        else:
            self.waiting_counter = 0

        if not all([np.isinf(v) for v in Q_state[1:]]) and self.waiting_counter > self.max_waiting:
            print(f"[QTablePolicy] on state {state} max waiting time exceeded")
            Q_state[action] = np.inf
            action = np.argmin(Q_state)

        self.prev_state = state
        self.prev_action = action
        return action
