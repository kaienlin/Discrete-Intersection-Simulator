import random
import sys, time, os
import numpy as np
import matplotlib.pyplot as plt

from simulator.intersection import Intersection
from simulator.simulation import Simulator
from environment.env import GraphBasedSimEnv

def get_4cz_intersection():
    '''
                   N
            ---------------
            |  1   |  2   |
            |      |      |
        W   ---------------   E
            |  3   |  4   |
            |      |      |
            ---------------
                   S
    '''
    I = Intersection()
    for cz_id in range(1, 5):
        I.add_conflict_zone(str(cz_id))
    
    I.add_src_lane("N", ["1"])
    I.add_src_lane("E", ["2"])
    I.add_src_lane("W", ["3"])
    I.add_src_lane("S", ["4"])

    I.add_dst_lane("N", ["2"])
    I.add_dst_lane("E", ["4"])
    I.add_dst_lane("W", ["1"])
    I.add_dst_lane("S", ["3"])

    I.add_trajectory(("1"))
    I.add_trajectory(("1", "3"))
    I.add_trajectory(("1", "3", "4"))

    I.add_trajectory(("2"))
    I.add_trajectory(("2", "1"))
    I.add_trajectory(("2", "1", "3"))

    I.add_trajectory(("3"))
    I.add_trajectory(("3", "4"))
    I.add_trajectory(("3", "4", "2"))

    I.add_trajectory(("4"))
    I.add_trajectory(("4", "2"))
    I.add_trajectory(("4", "2", "1"))

    return I

def add_random_traffic(sim: Simulator, max_time=300, max_vehicle_num=8, p=0.5):
    traj_per_src = {
        "N": {
            ("1"): "W",
            ("1", "3"): "S",
            ("1", "3", "4"): "E"
        },
        "E": {
            ("2"): "N",
            ("2", "1"): "W",
            ("2", "1", "3"): "S"
        },
        "W": {
            ("3"): "S",
            ("3", "4"): "E",
            ("3", "4", "2"): "N"
        },
        "S": {
            ("4"): "E",
            ("4", "2"): "N",
            ("4", "2", "1"): "W"
        }
    }
    veh_num = 0
    for t in range(max_time):
        for src_lane_id in sim.intersection.src_lanes:
            if random.random() < p and veh_num < max_vehicle_num:
                traj = random.choice(list(traj_per_src[src_lane_id].keys()))
                dst_lane_id = traj_per_src[src_lane_id][traj]
                sim.add_vehicle(f"vehicle-{veh_num}", t, traj, src_lane_id, dst_lane_id)
                veh_num += 1

def get_new_simulator():
    intersection = get_4cz_intersection()
    sim = Simulator(intersection)
    add_random_traffic(sim, max_vehicle_num=random.randint(1, 8))
    return sim

def load_Q_table(env):
    if os.path.exists("./Q.npy"):
        return np.load("./Q.npy")
    else:
        return np.zeros((env.observation_space.n, env.action_space.n))

def Q_learning():
    # Hyperparameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.5
    num_training_epoch = 1000
    num_evaluation_epoch = 100

    # create simulator and environment
    sim = get_new_simulator()
    env = GraphBasedSimEnv(sim)

    Q = load_Q_table(env)
    seen_state = set()
    for epoch in range(num_training_epoch):
        print(f"epoch = {epoch}: {len(seen_state)} / {env.observation_space.n} states explored")
        done = False
        state = env.reset()
        seen_state.add(state)
        cost_sum = 0
        
        while not done:
            # epsilon-greedy
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, env.action_space_size - 1)
            else:
                action = np.argmin(Q[state])

            # take action
            next_state, cost, done, _ = env.step(action)
            cost_sum += cost

            # update Q table
            next_min = np.min(Q[next_state])
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (cost + gamma * next_min)

            state = next_state
            seen_state.add(state)
        
        print(f"epoch: {epoch}, waiting time sum = {cost_sum}")
        
        if (epoch + 1) % 10 == 0:
            sim = get_new_simulator()
            env = GraphBasedSimEnv(sim)

    unexplored_state = random.choice(list(set(range(env.observation_space.n)).difference(seen_state)))
    print(env.decode_state(unexplored_state))
    sys.exit(0)

    # Evaluation
    cost_history = []
    for epoch in range(num_evaluation_epoch):
        sim = get_new_simulator()
        env = GraphBasedSimEnv(sim)

        done = False
        state = env.reset()
        cost_sum = 0
        
        while not done:
            env.render()
            action = np.argmin(Q[state])

            # Greedy Scheduler
            #action = 0
            #dec = env.decode_state(state)
            #print(dec["vehicle_positions"])
            #for pos, next in dec["vehicle_positions"].items():
            #    if "(SRC)" in pos:
            #        action = env.encode_action(pos[5:], is_cz=False)
            #        print(pos[5:])
            #        break
            #    elif next["waiting"]:
            #        action = env.encode_action(pos, is_cz=True)
            #        print(pos)
            #        break
                    
            env.render()
            state, cost, done, _ = env.step(action)
            cost_sum += cost
        
        cost_history.append(cost_sum)

    plt.plot(np.array(range(0, num_evaluation_epoch)), np.array(cost_history))
    plt.show()


if __name__ == "__main__":
    seed = 121
    random.seed(seed)
    np.random.seed(seed)
    intersection = get_4cz_intersection()
    Q_learning()

