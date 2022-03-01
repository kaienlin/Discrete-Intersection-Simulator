import numpy as np

from .base import Policy
from .greedy import IGreedyPolicy

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
        action = np.argmin(Q_state)
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
