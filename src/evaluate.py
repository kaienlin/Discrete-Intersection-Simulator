from environment import GraphBasedSimEnv
from simulator import Simulator, Intersection
from utility import read_intersection_from_json
import traffic_gen
import policy

from typing import Iterable
import random
import numpy as np
import fire

def evaluate(policy: policy.Policy, env: GraphBasedSimEnv):
    '''
    return the average waiting time of vehicles in seconds
    '''
    done = False
    state = env.reset()
    waiting_time_sum = 0
    while not done:
        action = policy.decide(state)
        state, cost, done, _ = env.step(action)
        waiting_time_sum += cost
    return waiting_time_sum / 10

def main(
    intersection_file_path: str,
    traffic_data_dir: str,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    intersection: Intersection = read_intersection_from_json(intersection_file_path)
    sim_gen: Iterable[Simulator] = traffic_gen.datadir_traffic_generator(intersection, traffic_data_dir)
    env = GraphBasedSimEnv(Simulator(intersection))

    # Modify this list to compare different policies
    policies = [
        ("iGreedy", policy.IGreedyPolicy(env)),
        ("DP", policy.QTablePolicy(np.load("DP.npy")))
    ]

    cost = [list() for _ in policies]
    for sim in sim_gen:
        env.reset(new_sim=sim)

        for i, (pi_name, pi) in enumerate(policies):
            c = evaluate(pi, env)
            cost[i].append(c)
            print(f"{pi_name}: {c}")
        
        print("-------")
    
    print("=== Average ===")
    for i, (pi_name, _) in enumerate(policies):
        print(f"{pi_name}: {sum(cost[i]) / len(cost[i])}")


if __name__ == "__main__":
    fire.Fire(main)
