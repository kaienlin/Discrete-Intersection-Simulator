import numpy as np

from .base import Policy

class QTablePolicy(Policy):
    def __init__(self, Q):
        self.Q = Q

    def decide(self, state: int) -> int:
        return np.argmin(self.Q[state])
