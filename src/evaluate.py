from typing import Iterable, Union
from pathlib import Path
import random
import pickle

from tqdm import tqdm
import numpy as np
import fire

from environment.tabular import position_based, vehicle_based
from simulation import Simulator, Intersection
from utility import read_intersection_from_json
from CP import solve_by_CP
import traffic_gen
import policy


def evaluate(P: policy.Policy, env: Union[position_based.SimulatorEnv, vehicle_based.SimulatorEnv]):
    '''
    return the average waiting time of vehicles in seconds
    '''
    if len(env.sim.vehicles) == 0:
        return 0

    done = False
    state = env.reset()
    waiting_time_sum = 0
    discount_factor = 1
    i = 0
    while not done:
        action = P.decide(state)
        state, cost, done, _ = env.step(action)
        waiting_time_sum += (discount_factor ** i) * cost
        i += 1
    return waiting_time_sum / 10 / len(env.sim.vehicles), waiting_time_sum >= env.DEADLOCK_COST


def batch_evaluate(
    P: policy.Policy,
    env: Union[position_based.SimulatorEnv, vehicle_based.SimulatorEnv],
    sim_gen: Iterable[Simulator]
):
    '''
    evaluate a batch (iterable of simulators) of traffic a time and return the average cost
    '''
    c_list = []
    for sim in sim_gen:
        env.reset(new_sim=sim)
        c = evaluate(P, env)
        c_list.append(c)
    return sum(c_list) / len(c_list)


def evaluate_ts(P, env):
    if len(env.sim.vehicles) == 0:
        return 0

    time_step = env.reset()
    cumulative_reward = 0
    timeout = 10000
    cnt = 0
    while not time_step.is_last():
        action = P.action(time_step)
        time_step = env.step(action)
        cumulative_reward += time_step.reward
        cnt += 1
        if cnt > timeout:
            print("TIMEOUT")
            break

    return cumulative_reward / 10 / len(env.sim.vehicles)


def batch_evaluate_ts(P, env, sim_gen):
    c_list = []
    for sim in sim_gen:
        env.sim = sim
        c = evaluate_ts(P, env)
        c_list.append(c)
    return sum(c_list) / len(c_list)


def main(
    intersection_file_path: str,
    traffic_data_dir: str,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)

    intersection: Intersection = read_intersection_from_json(
        intersection_file_path)
    sim_gen: Iterable[Simulator] = traffic_gen.datadir_traffic_generator(
        intersection, traffic_data_dir)

    checkpoint_path = Path("checkpoints/reduced-2x2-8v-1q/")
    env = vehicle_based.SimulatorEnv(Simulator(intersection))

    with open(checkpoint_path / "env.p", "rb") as f:
        _env = pickle.load(f)
        env.encoding_table = _env.encoding_table
        env.decoding_table = _env.decoding_table
        del _env
    
    env.reset(new_sim=Simulator(intersection))

    # Modify this list to compare different policies
    policies = [
        ("iGreedy", policy.IGreedyPolicy(env)),
        ("Q-learning", policy.QTablePolicy(env, np.load(checkpoint_path / "Q.npy"))),
    ]

    cost = [[] for _ in policies]
    cost.append([])
    deadlock_cnt = [0 for _ in policies]
    deadlock_cnt.append(0)
    pbar = tqdm()
    for sim in sim_gen:
        env.reset(new_sim=sim)
        optimum = solve_by_CP(sim)
        cost[-1].append(optimum)

        for i, (pi_name, pi) in enumerate(policies):
            c, deadlock = evaluate(pi, env)
            #print(pi_name, c)
            # if round(c, 2) < round(optimum, 2):
            #    print(c, optimum)
            if deadlock:
                deadlock_cnt[i] += 1
            else:
                cost[i].append(c)
        #print("------")

        pbar.update(1)

    policies.append(("Optimal", ""))
    print("=== Average ===")
    for i, (pi_name, _) in enumerate(policies):
        print(
            f"{pi_name}: {sum(cost[i]) / len(cost[i])}; {deadlock_cnt[i]} / {len(cost[i])}")


if __name__ == "__main__":
    fire.Fire(main)
