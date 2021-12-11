import random
import itertools
from typing import Dict, Set

from simulator import Intersection, Simulator

def get_src_traj_dict(intersection: Intersection):
    traj_per_src = {}
    src_lanes: Dict[str, Set[str]] = intersection.src_lanes
    for src_lane_id in src_lanes:
        traj_per_src[src_lane_id] = {}
    for traj in intersection.trajectories:
        for src_lane_id, associated_czs in src_lanes.items():
            if traj[0] in associated_czs:
                dst_lane_id = intersection.get_traj_dst_lane(traj)
                traj_per_src[src_lane_id][traj] = dst_lane_id
    return traj_per_src

def add_random_traffic(sim: Simulator, max_time=300, max_vehicle_num=8, p=0.5):
    src_to_traj = get_src_traj_dict(sim.intersection)
    veh_num = 0
    for t in range(max_time):
        for src_lane_id in sim.intersection.src_lanes:
            if random.random() < p and veh_num < max_vehicle_num:
                traj = random.choice(list(src_to_traj[src_lane_id].keys()))
                dst_lane_id = src_to_traj[src_lane_id][traj]
                sim.add_vehicle(f"vehicle-{veh_num}", t, traj, src_lane_id, dst_lane_id)
                veh_num += 1

def random_traffic_generator(intersection: Intersection, num_iter: int = 10000):
    for _ in range(num_iter):
        sim = Simulator(intersection)
        add_random_traffic(sim, max_vehicle_num=random.randint(1, 8))
        yield sim

def enumerate_traffic_patterns_generator(intersection: Intersection):
    src_to_traj = get_src_traj_dict(intersection)
    src_lane_id_list = []
    S = []
    n = 1
    for src_lane_id, trajs in src_to_traj.items():
        S_i = []
        for num in range(len(intersection.conflict_zones)-1, -1, -1):
            S_i.extend(itertools.product(trajs, repeat=num))
        S.append(S_i)
        n *= len(S_i)
        src_lane_id_list.append(src_lane_id)

    for traffic in itertools.product(*S):
        sim = Simulator(intersection)
        veh_cnt = 0
        for src_lane_id, traj_list in zip(src_lane_id_list, traffic):
            for traj in traj_list:
                dst_lane_id = src_to_traj[src_lane_id][traj]
                sim.add_vehicle(f"vehicle-{veh_cnt}", 0, traj, src_lane_id, dst_lane_id)
                veh_cnt += 1
        yield sim
    
           