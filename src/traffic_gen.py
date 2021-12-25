import random
from typing import Dict, Set
from pathlib import Path

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

def add_random_traffic(sim: Simulator, max_time=300, max_vehicle_num=8, p=0.05):
    src_to_traj = get_src_traj_dict(sim.intersection)
    veh_num = 0
    for t in range(max_time):
        for src_lane_id in sim.intersection.src_lanes:
            if random.random() < p and veh_num < max_vehicle_num:
                traj = random.choice(list(src_to_traj[src_lane_id].keys()))
                dst_lane_id = src_to_traj[src_lane_id][traj]
                sim.add_vehicle(f"vehicle-{veh_num}", t, traj, src_lane_id, dst_lane_id)
                veh_num += 1

def random_traffic_generator(
    intersection: Intersection,
    num_iter: int = 10000,
    max_vehicle_num: int = 100,
    poisson_parameter_list = [0.1, 0.3, 0.5, 0.7, 0.9]
):
    cond = lambda _: True
    if num_iter > 0:
        cond = lambda i: i < num_iter
    i = 0
    while cond(i):
        sim = Simulator(intersection)
        add_random_traffic(sim, max_vehicle_num=max_vehicle_num,
                           p=random.choice(poisson_parameter_list) / 10)
        i += 1
        yield sim

def datadir_traffic_generator(intersection: Intersection, data_dir):
    data_dir = Path(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise Exception("data_dir is not a directory")

    for traffic_file in data_dir.iterdir():
        sim = Simulator(intersection)
        sim.load_traffic(traffic_file)
        yield sim


if __name__ == "__main__":
    from utility import get_4cz_intersection
    import numpy as np

    data_dir = Path("./validation/4cz/")
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    intersection = get_4cz_intersection()
    random.seed(0)
    gen = random_traffic_generator(intersection, num_iter=10, poisson_parameter_list=np.arange(0, 1, 0.01).tolist())
    for i, sim in enumerate(gen):
        sim.dump_traffic(data_dir / f"{i}.json")
