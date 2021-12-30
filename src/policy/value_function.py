import numpy as np

from .base import Policy

class QTablePolicy(Policy):
    def __init__(self, Q):
        self.Q = Q
        
        self.prev_state: int = -1
        self.prev_action: int = -1
        self.waiting_counter: int = 0
        self.max_waiting = 100

    def decide(self, state: int) -> int:
        Q_state = self.Q[state]
        action = np.argmin(Q_state)
        if action == 0:
            if state == self.prev_state:
                self.waiting_counter += 1
            else:
                self.waiting_counter = 1
        else:
            self.waiting_counter = 0

        if self.waiting_counter > self.max_waiting:
            print(f"on state {state} max waiting time exceeded")
            Q_state[action] = np.inf
            action = np.argmin(Q_state)

        self.prev_state = state
        self.prev_action = action
        return action
