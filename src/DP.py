import random
import numpy as np
from tqdm import tqdm

from environment import ProbabilisticEnv
from utility import read_intersection_from_json

def value_iteration(env: ProbabilisticEnv, theta=1e-3, discount_factor=0.95):
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

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    intersection = read_intersection_from_json("../intersection_configs/1x1.json")
    env = ProbabilisticEnv(intersection)
    DP_policy = value_iteration(env)

    np.save("./DP.npy", DP_policy)
