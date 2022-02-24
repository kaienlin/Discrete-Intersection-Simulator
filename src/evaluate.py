from environment import position_based, vehicle_based
from simulator import Simulator, Intersection, vehicle
from utility import read_intersection_from_json
import traffic_gen
import policy

from typing import Iterable, Union
from tqdm import tqdm
import random
import numpy as np
import fire
import pickle

def evaluate(P: policy.Policy, env: Union[position_based.SimulatorEnv, vehicle_based.SimulatorEnv]):
    '''
    return the average waiting time of vehicles in seconds
    '''
    done = False
    state = env.reset()
    waiting_time_sum = 0
    while not done:
        action = P.decide(state)
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
    #env = vehicle_based.SimulatorEnv(Simulator(intersection))
    env = pickle.load(open("disturbed2x2/env.p", "rb"))
    env.reset(new_sim=Simulator(intersection))

    # Modify this list to compare different policies
    policies = [
        ("iGreedy", policy.IGreedyPolicy(env)),
        ("Q-learning", policy.QTablePolicy(env, np.load("disturbed2x2/Q.npy")))
    ]

    cost = [list() for _ in policies]
    deadlock_cnt = [0 for _ in policies]
    pbar = tqdm()
    for sim in sim_gen:
        env.reset(new_sim=sim)

        for i, (pi_name, pi) in enumerate(policies):
            c = evaluate(pi, env)
            cost[i].append(c)
            if c > 1e6:
                deadlock_cnt[i] += 1
        
        pbar.update(1)
    
    print("=== Average ===")
    for i, (pi_name, _) in enumerate(policies):
        print(f"{pi_name}: {sum(cost[i]) / len(cost[i])}; {deadlock_cnt[i]} / {len(cost[i])}")


if __name__ == "__main__":
    fire.Fire(main)
