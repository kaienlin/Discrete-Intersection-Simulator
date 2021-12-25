import sys, random, os, pickle
from typing import Iterable
import numpy as np

from environment import GraphBasedSimEnv
from simulator import Simulator

from utility import get_4cz_intersection
from traffic_gen import random_traffic_generator

def load_Q_table(env):
    if os.path.exists("./Q.npy"):
        return np.load("./Q.npy")
    else:
        return np.zeros((env.observation_space.n, env.action_space.n))

def save_Q_table(Q):
    np.save("./Q.npy", Q)

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
    
def Q_learning(simulator_generator: Iterable[Simulator], epoch_per_traffic=100):
    # create simulator and environment
    sim = next(simulator_generator)
    env = GraphBasedSimEnv(sim)

    num_actable_states = 0
    for s in range(env.state_space_size):
        if env.is_actable_state(s):
            num_actable_states += 1
    print(f"number of actable states = {num_actable_states}") 

    Q = load_Q_table(env)
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

        if (epoch + 1) % 10000 == 0:
            save_Q_table(Q)
            pickle.dump(seen_state, open("seen.p", "wb"))
        epoch += 1

    save_Q_table(Q)
    pickle.dump(seen_state, open("seen.p", "wb"))


if __name__ == "__main__":
    seed = 12245
    random.seed(seed)
    np.random.seed(seed)
    intersection = get_4cz_intersection()
    sim_gen = random_traffic_generator(intersection, num_iter=0)
    Q_learning(sim_gen)
