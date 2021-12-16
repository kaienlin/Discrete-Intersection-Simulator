import math
import sys, pprint, random
import numpy as np
from tqdm import tqdm

from environment import ProbabilisticEnv
from simulator import intersection
from utility import get_4cz_intersection

def value_iteration(env: ProbabilisticEnv, theta=0.01, discount_factor=1.0):
    def one_step_lookahead(state, V):
        A = np.zeros(env.action_space_size)
        for a in range(env.action_space_size):
            for prob, next_state, cost in env.get_transitions(state, a):
                A[a] += prob * (cost + discount_factor * V[next_state])
        return A
    
    V = np.zeros(env.state_space_size)
    epoch = 0
    delta = theta + 1.
    while delta > theta:
        epoch += 1
        delta = 0.0
        for s in tqdm(range(env.state_space_size), leave=False, ascii=True):
            A = one_step_lookahead(s, V)
            best_action_value = np.min(A)
            delta = max(delta, np.abs(V[s] - best_action_value))
            V[s] = best_action_value
        print(f"epoch {epoch}: delta = {delta}")

    policy = np.zeros((env.state_space_size, env.action_space_size))
    for s in range(env.state_space_size):
        A = one_step_lookahead(s, V)
        best_action = np.argmin(A)
        policy[s, best_action] = 1.0

    return policy


if __name__ == "__main__":
    intersection = get_4cz_intersection()
    env = ProbabilisticEnv(intersection)
    value_iteration(env)
