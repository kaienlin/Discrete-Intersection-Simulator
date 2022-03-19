from typing import Any, Iterable, Dict, Optional, List
from pathlib import Path
from collections import defaultdict
import os
import sys
import random
import pickle
import fcntl

from tqdm import tqdm
import numpy as np
import fire

from simulation import Simulator, Intersection
from utility import read_intersection_from_json, DynamicQtable, FileLock
from evaluate import batch_evaluate
from policy import QTablePolicy
import traffic_gen
import environment


def load_Q_table(env, path):
    table = DynamicQtable(env.action_space_size, init_state_num=1<<20)
    if os.path.exists(path):
        with FileLock(path, "shared"):
            table.load(path)
    return table


def save_Q_table(Q: DynamicQtable, path):
    with FileLock(path, "exclusive"):
        Q.save(path)


def train_Q(
    env: environment.vehicle_based.SimulatorEnv,
    Q: DynamicQtable,
    seen_state: Optional[Dict] = None,
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

        # take action
        state, cost, done, _ = env.step(action)

    for S_0, env_s in env.get_snapshots():
        done = False
        state = S_0
        while not done:
            effective_actions = [a for a in range(env_s.action_space_size)
                                if env_s.is_effective_action_of_state(a, state)]

            # update state seen count
            if seen_state is not None and len(effective_actions) > 1:
                seen_state[state] += 1

            # epsilon-greedy
            if random.uniform(0, 1) < epsilon:
                action = random.choice(effective_actions)
            else:
                action = effective_actions[Q[state][effective_actions].argmin()]
            
            next_state, cost, done, _ = env_s.step(action)

            # update Q table
            next_min = np.min(Q[next_state])
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (cost + gamma * next_min)

            # ensure the Q values of invalid state-action pairs are np.inf
            for a in range(env.action_space_size):
                if a not in effective_actions:
                    Q[state, a] = np.inf

            state = next_state


def synthesize(
    env,
    Q: DynamicQtable,
    seen_state: Optional[Dict] = None,
    alpha: float = 0.1,
    gamma: float = 0.9,
    traj_file_list: List[str] = []
):
    for traj_file in traj_file_list:
        path = Path(traj_file)
        if not path.is_file():
            continue

        with FileLock(path, "exclusive"):
            with open(path, "rt") as f:
                for traj in f.readlines():
                    traj = traj.split()
                    traj = [[int(v) for v in traj[i:i+3]] for i in range(0, len(traj), 3)]

                    for i, (state, action, cost) in enumerate(traj[:-1]):
                        effective_actions = [a for a in range(env.action_space_size)
                                            if env.is_effective_action_of_state(a, state)]

                        if seen_state and len(effective_actions) > 1:
                            seen_state[state] += 1

                        next_state = traj[i+1][0]
                        next_min = np.min(Q[next_state])
                        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (cost + gamma * next_min)

                        for a in range(env.action_space_size):
                            if a not in effective_actions:
                                Q[state, a] = np.inf

            # Clear the file content
            with open(path, "wt"):
                pass


def Q_learning(
    simulator_generator: Iterable[Simulator],
    checkpoint_path: Path,
    max_vehicle_num: int = 8,
    max_vehicle_num_per_src_lane: int = 1,
    epoch_per_traffic: int = 10,
    epoch_per_checkpoint: int = 10000,
    eval_data_dir: Optional[str] = None,
    alpha: float = 0.01,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    traj_file_list: List[str] = []
):
    # create simulator and environment
    sim = next(simulator_generator)

    enc_dec_table_path: Path = checkpoint_path / "enc_dec_table.p"
    Q_table_path: Path = checkpoint_path / "Q.npy"
    seen_path: Path = checkpoint_path / "seen.p"

    best_Q_table_path: Path = checkpoint_path / "Q.best.npy"

    env = environment.vehicle_based.SimulatorEnv(sim, 
            max_vehicle_num=max_vehicle_num, max_vehicle_num_per_src_lane=max_vehicle_num_per_src_lane)

    if enc_dec_table_path.is_file():
        env.load_enc_dec_tables(enc_dec_table_path)

    Q = load_Q_table(env, Q_table_path)
    seen_state = defaultdict(int)
    if seen_path.is_file():
        with open(seen_path, "rb") as f:
            seen_state = pickle.load(f)

    def evaluate(env, Q) -> float:
        sim_gen: Iterable[Simulator] = traffic_gen.datadir_traffic_generator(env.sim.intersection, eval_data_dir)
        P = QTablePolicy(env, Q)
        return batch_evaluate(P, env, sim_gen)

    epoch = 0
    best_performance = int(1e9)
    pbar = tqdm()
    while True:
        pbar.set_description(
            f"epoch = {epoch}: {len(seen_state)} / {len(env.decoding_table)} states explored")
        train_Q(env, Q, seen_state, alpha=alpha, gamma=gamma, epsilon=epsilon)

        if (epoch + 1) % epoch_per_traffic == 0:
            try:
                sim = next(simulator_generator)
            except StopIteration:
                break
            env.reset(sim)

        if (epoch + 1) % epoch_per_checkpoint == 0:
            synthesize(env, Q, seen_state, alpha=alpha, gamma=gamma, traj_file_list=traj_file_list)
            pbar.set_description("Saving...")
            save_Q_table(Q, Q_table_path)
            pickle.dump(seen_state, open(seen_path, "wb"))

            if eval_data_dir is not None:
                performance = evaluate(env, Q)
                if performance < best_performance:
                    save_Q_table(Q, best_Q_table_path)
                    best_performance = performance

        epoch += 1


def explore_Q(
    env: environment.vehicle_based.SimulatorEnv,
    Q: DynamicQtable,
    trajectories_record_file: Optional[Path] = None,
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

        # take action
        state, cost, done, _ = env.step(action)

    for S_0, env_s in env.get_snapshots():
        done = False
        state = S_0
        trajectory = []
        while not done:
            effective_actions = [a for a in range(env_s.action_space_size)
                                if env_s.is_effective_action_of_state(a, state)]

            # epsilon-greedy
            if random.uniform(0, 1) < epsilon:
                action = random.choice(effective_actions)
            else:
                action = effective_actions[Q[state][effective_actions].argmin()]
            
            next_state, cost, done, _ = env_s.step(action)
            trajectory.append([state, action, cost])

            state = next_state

        trajectory.append([state, 0, 0])
        with FileLock(trajectories_record_file, "exclusive"):
            with open(trajectories_record_file, "at") as f:
                for s, a, c in trajectory:
                    f.write(f"{s} {a} {c} ")
                f.write("\n")


def exploration_only(
    simulator_generator: Iterable[Simulator],
    checkpoint_path: Path,
    max_vehicle_num: int = 8,
    max_vehicle_num_per_src_lane: int = 1,
    epsilon: float = 0.1,
    epoch_per_traffic: int = 10,
    epoch_per_checkpoint: int = 10000,
    trajectories_record_file: Optional[Path] = None
):
    # create simulator and environment
    sim = next(simulator_generator)

    enc_dec_table_path: Path = checkpoint_path / "enc_dec_table.p"
    Q_table_path: Path = checkpoint_path / "Q.npy"

    env = environment.vehicle_based.SimulatorEnv(sim, 
            max_vehicle_num=max_vehicle_num, max_vehicle_num_per_src_lane=max_vehicle_num_per_src_lane)

    if enc_dec_table_path.is_file():
        env.load_enc_dec_tables(enc_dec_table_path)

    Q = load_Q_table(env, Q_table_path)

    epoch = 0
    pbar = tqdm()
    while True:
        pbar.set_description(f"epoch = {epoch}")
        explore_Q(env, Q, epsilon=epsilon, trajectories_record_file=trajectories_record_file)

        if (epoch + 1) % epoch_per_traffic == 0:
            try:
                sim = next(simulator_generator)
            except StopIteration:
                break
            env.reset(sim)

        if (epoch + 1) % epoch_per_checkpoint == 0:
            pbar.set_description("Saving...")
            load_Q_table(env, Q_table_path)

        epoch += 1
