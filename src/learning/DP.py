import random
import numpy as np
from tqdm import tqdm
import fire

import environment
from simulation import Intersection
from utility import read_intersection_from_json

def value_iteration(env: environment.tabular.position_based.ProbabilisticEnv, theta=1e-3, discount_factor=0.95):
    def one_step_lookahead(state, V):
        A = np.zeros(env.action_space_size)
        effective_actions = [a for a in range(env.action_space_size) if env.is_effective_action_of_state(a, state)]
        for a in effective_actions:
            for prob, next_state, cost in env.get_transitions(state, a):
                A[a] += prob * (cost + discount_factor * V[next_state])
        for a in range(env.action_space_size):
            if a not in effective_actions:
                A[a] = np.inf
        return A
    
    V = np.zeros(env.state_space_size)
    epoch = 0
    delta = theta + 1.
    while delta > theta:
        epoch += 1
        delta = 0.0
        for s in tqdm(range(env.state_space_size), ncols=70, leave=False, ascii=True):
            A = one_step_lookahead(s, V)
            best_action_value = np.min(A)
            delta = max(delta, np.abs(V[s] - best_action_value))
            V[s] = best_action_value
        print(f"epoch {epoch}: delta = {delta}")

    Q = np.zeros((env.state_space_size, env.action_space_size))
    for s in range(env.state_space_size):
        A = one_step_lookahead(s, V)
        Q[s] = A

    return Q

def main(
    intersection_file_path: str,
    seed: int = 0,
    Q_table_path: str = "DP.npy",
    theta: float = 1e-3,
    discount_factor: float = 0.95,
):
    intersection: Intersection = read_intersection_from_json(intersection_file_path)
    random.seed(seed)
    np.random.seed(seed)
    env = environment.tabular.position_based.ProbabilisticEnv(intersection)
    DP_policy = value_iteration(env, theta=theta, discount_factor=discount_factor)
    np.save(Q_table_path, DP_policy)


if __name__ == "__main__":
    fire.Fire(main)
