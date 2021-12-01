from environment.env import GraphBasedSimEnv
import simulator
import random
import sys, time
import numpy as np
from simulator.simulation import Simulator

from simulator.vehicle import VehicleState

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
    I = simulator.Intersection()
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

def add_random_traffic(sim: Simulator, max_time=300, max_vehicle_num=10, p=0.1):
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

if __name__ == "__main__":
    seed = 121
    random.seed(seed)
    np.random.seed(seed)
    intersection = get_4cz_intersection()
    sim = simulator.Simulator(intersection)
    add_random_traffic(sim)

    env = GraphBasedSimEnv(sim)

    # Hyperparameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1

    Qtable = np.zeros((env.observation_space.n, env.action_space.n))
    state = env.reset()
    done = False
    while not done:
        env.render()
        #time.sleep(1)
        action = random.randint(0, env.action_space_size - 1)
        next_state, cost, done, _ = env.step(action)

    env.render()
    env.sim.print_TCG()
    print(env.sim.status)