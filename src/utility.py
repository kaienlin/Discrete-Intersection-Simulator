import random

from simulator import Intersection
from simulator import Simulator

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
