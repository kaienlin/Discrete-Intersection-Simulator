from typing import Any, Iterable, Dict, Optional
from pathlib import Path
from collections import defaultdict
import os
import sys
import random
import pickle

from tqdm import tqdm
import numpy as np
import fire

from simulation import Simulator, Intersection
from utility import read_intersection_from_json, DynamicQtable
from evaluate import batch_evaluate
from policy import QTablePolicy
import traffic_gen
import environment

def load_Q_table(env, path):
    table = DynamicQtable(env.action_space_size, init_state_num=1<<20)
    if os.path.exists(path):
        table.load(path)
    return table


def save_Q_table(Q: DynamicQtable, path):
    Q.save(path)


def train_Q(
        env: environment.vehicle_based.SimulatorEnv,
        Q: DynamicQtable,
        seen_state: Optional[Dict] = None,
        prob_env: Optional[Any] = None,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
    ):
    done = False
    state = env.reset()

    while not done:
        effective_actions = [a for a in range(env.action_space_size)
                             if env.is_effective_action_of_state(a, state)]
        # epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = random.choice(effective_actions)
        else:
            action = effective_actions[Q[state][effective_actions].argmin()]

        if seen_state is not None and len(effective_actions) > 1:
            seen_state[state] += 1

        # take action
        next_state, cost, done, _ = env.step(action)

        # check reachability
        if prob_env is not None and not prob_env.reachable(state, action, next_state):
            print("!!!!!!!!!!!!!!!!!!!!")
            print("Invalid state transition")
            print("* Source:")
            env.decode_state(state).print()
            print("* Action:")
            env.decode_action(action).print()
            print("* Destination:")
            env.decode_state(next_state).print()
            print("!!!!!!!!!!!!!!!!!!!!")
            sys.exit(0)

        # update Q table
        next_min = np.min(Q[next_state])
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (cost + gamma * next_min)
        for a in range(env.action_space_size):
            if a not in effective_actions:
                Q[state, a] = np.inf

        state = next_state


def Q_learning(
    simulator_generator: Iterable[Simulator],
    checkpoint_path: Path,
    max_vehicle_num: int = 8,
    max_vehicle_num_per_src_lane: int = 1,
    epoch_per_traffic: int = 10,
    epoch_per_checkpoint: int = 10000,
    eval_data_dir: Optional[str] = None
):
    # create simulator and environment
    sim = next(simulator_generator)

    env_path: Path = checkpoint_path / "env.p"
    Q_table_path: Path = checkpoint_path / "Q.npy"
    seen_path: Path = checkpoint_path / "seen.p"

    best_Q_table_path: Path = checkpoint_path / "Q.best.npy"

    if env_path.is_file():
        env = pickle.load(open(env_path, "rb"))
    else:
        env = environment.vehicle_based.SimulatorEnv(sim, 
            max_vehicle_num=max_vehicle_num, max_vehicle_num_per_src_lane=max_vehicle_num_per_src_lane)

    num_actable_states = 0
    for s in range(env.state_space_size):
        if env.is_actable_state(s):
            num_actable_states += 1
    print(f"number of actable states = {num_actable_states}")

    Q = load_Q_table(env, Q_table_path)
    seen_state = pickle.load(open(seen_path, "rb")) if seen_path.is_file() else defaultdict(int)

    for s in range(env.state_space_size):
        for a in range(env.action_space_size):
            if not env.is_effective_action_of_state(a, s):
                Q[s][a] = np.inf

    def evaluate(env, Q) -> float:
        sim_gen: Iterable[Simulator] = traffic_gen.datadir_traffic_generator(env.sim.intersection, eval_data_dir)
        P = QTablePolicy(env, Q)
        return batch_evaluate(P, env, sim_gen)

    epoch = 0
    best_performance = 1e9
    pbar = tqdm()
    while True:
        pbar.set_description(
            f"epoch = {epoch}: {len(seen_state)} states explored, best performance = {best_performance}")
        train_Q(env, Q, seen_state, prob_env=None)

        if (epoch + 1) % epoch_per_traffic == 0:
            try:
                sim = next(simulator_generator)
            except StopIteration:
                break
            env.reset(sim)

        if (epoch + 1) % epoch_per_checkpoint == 0:
            pbar.set_description("Saving...")
            save_Q_table(Q, Q_table_path)
            pickle.dump(seen_state, open(seen_path, "wb"))
            pickle.dump(env, open(env_path, "wb"))
            if eval_data_dir is not None:
                performance = evaluate(env, Q)
                if performance < best_performance:
                    save_Q_table(Q, best_Q_table_path)
                    best_performance = performance
        epoch += 1

    print("Saving...", end=" ")
    save_Q_table(Q, Q_table_path)
    pickle.dump(seen_state, open(seen_path, "wb"))
    pickle.dump(env, open(env_path, "wb"))
    print("...done")


def main(
    intersection_file_path: str,
    seed: int = 0,
    max_vehicle_num: int = 8,
    max_vehicle_num_per_src_lane: int = 1,
    traffic_generator_name: str = "random_traffic_generator",
    traffic_generator_kwargs: Dict = {},
    checkpoint_dir: str = "./",
    epoch_per_traffic: int = 10,
    epoch_per_checkpoint: int = 10000,
    eval_data_dir: Optional[str] = None
):
    intersection: Intersection = read_intersection_from_json(intersection_file_path)
    random.seed(seed)
    np.random.seed(seed)
    sim_gen: Iterable[Simulator] = getattr(traffic_gen, traffic_generator_name)(intersection, **traffic_generator_kwargs)

    checkpoint_dir_path: Path = Path(checkpoint_dir)
    if not checkpoint_dir_path.is_dir():
        checkpoint_dir_path.mkdir()

    Q_learning(sim_gen, checkpoint_dir_path,
               max_vehicle_num=max_vehicle_num, max_vehicle_num_per_src_lane=max_vehicle_num_per_src_lane,
               epoch_per_traffic=epoch_per_traffic, epoch_per_checkpoint=epoch_per_checkpoint, eval_data_dir=eval_data_dir)


if __name__ == "__main__":
    fire.Fire(main)
