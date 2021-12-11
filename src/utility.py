import json

from simulator import Intersection

def read_intersection_from_json(file_path):
    fp = open(file_path, "r")
    cfg = json.load(fp)
    I = Intersection()
    for cz_id in cfg["conflict_zones"]:
        I.add_conflict_zone(cz_id)

    for src_lane_id, info in cfg["source_lanes"].items():
        associated_czs = info["associated_conflict_zones"]
        I.add_src_lane(src_lane_id, associated_czs)

    for dst_lane_id, info in cfg["destination_lanes"].items():
        associated_czs = info["associated_conflict_zones"]
        I.add_dst_lane(dst_lane_id, associated_czs)

    for trajectory in cfg["trajectories"]:
        I.add_trajectory(tuple(trajectory))

    return I

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
    return read_intersection_from_json("../intersection_configs/2x2.json")

