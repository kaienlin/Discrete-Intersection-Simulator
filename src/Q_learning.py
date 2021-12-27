import sys, random, os, pickle
from typing import Generator, Iterable, Dict
import numpy as np
import fire

import traffic_gen
from environment import GraphBasedSimEnv
from simulator import Simulator, Intersection
from utility import read_intersection_from_json

def load_Q_table(env, path):
    if os.path.exists(path):
        return np.load(path)
    else:
        return np.zeros((env.observation_space.n, env.action_space.n))

def save_Q_table(Q, path):
    np.save(path, Q)

def train_Q(env: GraphBasedSimEnv, Q, seen_state=None, prob_env=None, alpha=0.1, gamma=1.0, epsilon=0.2):
    done = False
    state = env.reset()
    if seen_state is not None:
        seen_state.add(state)
    
    while not done:
        # epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, env.action_space_size - 1)
        else:
            action = np.argmin(Q[state])

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

        state = next_state
        if seen_state is not None:
            seen_state.add(state)
    
def Q_learning(
    simulator_generator: Iterable[Simulator],
    Q_table_path: str = "Q.npy",
    epoch_per_traffic: int = 10,
    epoch_per_checkpoint: int = 1000
):
    # create simulator and environment
    sim = next(simulator_generator)
    env = GraphBasedSimEnv(sim)

    num_actable_states = 0
    for s in range(env.state_space_size):
        if env.is_actable_state(s):
            num_actable_states += 1
    print(f"number of actable states = {num_actable_states}") 

    Q = load_Q_table(env, Q_table_path)
    seen_state = pickle.load(open("seen.p", "rb")) if os.path.exists("seen.p") else set()

    epoch = 0
    while True:
        print(f"epoch = {epoch}: {len(seen_state)} / {num_actable_states} states explored")
        train_Q(env, Q, seen_state, prob_env=None)
        
        if (epoch + 1) % epoch_per_traffic == 0:
            try:
                sim = next(simulator_generator)
            except StopIteration:
                break
            env.reset(sim)

        if (epoch + 1) % epoch_per_checkpoint == 0:
            save_Q_table(Q, Q_table_path)
            pickle.dump(seen_state, open("seen.p", "wb"))
        epoch += 1

    save_Q_table(Q, Q_table_path)
    pickle.dump(seen_state, open("seen.p", "wb"))

def main(
    intersection_file_path: str,
    seed: int = 0,
    traffic_generator_name: str = "random_traffic_generator",
    traffic_generator_kwargs: Dict = {},
    Q_table_path: str = "Q.npy",
    epoch_per_traffic: int = 10,
    epoch_per_checkpoint: int = 1000
):
    intersection: Intersection = read_intersection_from_json(intersection_file_path)
    random.seed(seed)
    np.random.seed(seed)
    sim_gen: Iterable[Simulator] = getattr(traffic_gen, traffic_generator_name)(intersection, **traffic_generator_kwargs)
    Q_learning(sim_gen, Q_table_path, epoch_per_traffic=epoch_per_traffic, epoch_per_checkpoint=epoch_per_checkpoint)


if __name__ == "__main__":
    fire.Fire(main)
