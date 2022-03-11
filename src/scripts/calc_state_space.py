from typing import List, Tuple, Literal, Optional
from dataclasses import dataclass, field
import copy

from utility import read_intersection_from_json
from simulation import Intersection
from traffic_gen import get_src_traj_dict

@dataclass(order=True, unsafe_hash=True)
class VehicleState:
    src_lane: str = ""
    trajectory: Tuple[str] = field(default_factory=tuple)
    position: int = -1
    state: Literal["waiting", "non-waiting"] = "non-waiting"


def gen_int_partitions(n: int, size: int) -> List[List[int]]:
    if size == 1:
        return [[n]]
    res = []
    for i in range(0, n+1):
        for p in gen_int_partitions(n - i, size - 1):
            res.append([i] + p)
    return res


def calculate_state_space_size(
    intersection: Intersection,
    max_vehicle_num: int,
    max_queue_length: Optional[int] = None
):
    if max_queue_length is None:
        max_queue_length = max_vehicle_num
    conflict_zones = sorted(intersection.conflict_zones)
    src_traj_dict = get_src_traj_dict(intersection)

    used_cz = set()
    waited_cz = set()
    possible_states = ("waiting", "non-waiting")
    res = []
    vehicle_list = []

    def fill_src_lane(idx: int, cur_vehicle_num: int, actable: bool) -> int:
        if cur_vehicle_num == max_vehicle_num:
            if actable:
                res.append(tuple(sorted(copy.deepcopy(vehicle_list))))
                return 1
            return 0
        if idx == len(src_traj_dict):
            if actable:
                res.append(tuple(sorted(copy.deepcopy(vehicle_list))))
                return 1
            return 0

        size = 0
        src_lane_id = sorted(src_traj_dict.keys())[idx]
        for n in range(1, min(max_queue_length + 1, max_vehicle_num - cur_vehicle_num + 1)):
            possible_trajs = sorted(src_traj_dict[src_lane_id])
            partitions = gen_int_partitions(n, len(possible_trajs))
            for p in partitions:
                new_vehicles = []
                for num, traj in zip(p, possible_trajs):
                    new_vehicles.extend([VehicleState(trajectory=traj) for _ in range(num)])
                for num, traj in zip(p, possible_trajs):
                    if num > 0 and traj[0] not in used_cz:
                        for v in new_vehicles:
                            if v.trajectory == traj:
                                v.state = "waiting"
                                break
                        vehicle_list.extend(new_vehicles)
                        size += fill_src_lane(idx + 1, cur_vehicle_num + n, True)
                        for _ in range(len(new_vehicles)):
                            vehicle_list.pop(-1)
                        for v in new_vehicles:
                            if v.trajectory == traj and v.state == "waiting":
                                v.state = "non-waiting"
                                break
                vehicle_list.extend(new_vehicles)
                size += fill_src_lane(idx + 1, cur_vehicle_num + n, actable)
                for _ in range(len(new_vehicles)):
                    vehicle_list.pop(-1)

        size += fill_src_lane(idx + 1, cur_vehicle_num, actable)

        return size


    def fill_cz(idx: int, cur_vehicle_num: int, actable: bool) -> int:
        if cur_vehicle_num == max_vehicle_num:
            if actable:
                res.append(tuple(sorted(copy.deepcopy(vehicle_list))))
                return 1
            return 0
        if idx == len(conflict_zones):
            return fill_src_lane(0, cur_vehicle_num, actable)
        if conflict_zones[idx] in waited_cz:
            return fill_cz(idx + 1, cur_vehicle_num, actable)

        size = 0
        cz_id = conflict_zones[idx]
        possible_trajs = set([traj[traj.index(cz_id):] for traj in intersection.trajectories if cz_id in traj])

        # fill this cz with one vehicle
        used_cz.add(cz_id)
        for traj in possible_trajs:
            traj = tuple(traj)
            next_cz = traj[traj.index(cz_id) + 1] if traj.index(cz_id) + 1 < len(traj) else "$"
            for state in possible_states:
                if state == "waiting":
                    if next_cz not in used_cz and next_cz != "$":
                        waited_cz.add(next_cz)
                        vehicle_list.append(VehicleState(trajectory=traj, position=0, state=state))
                        size += fill_cz(idx + 1, cur_vehicle_num + 1, True)
                        vehicle_list.pop(-1)
                        waited_cz.remove(next_cz)
                else:
                    vehicle_list.append(VehicleState(trajectory=traj, position=0, state=state))
                    size += fill_cz(idx + 1, cur_vehicle_num + 1, actable)
                    vehicle_list.pop(-1)
        used_cz.remove(cz_id)
        # not fill this cz with any vehicles
        size += fill_cz(idx + 1, cur_vehicle_num, actable)

        return size

    num_states = fill_cz(0, 0, False)

    return num_states


if __name__ == "__main__":
    intersection = read_intersection_from_json("../intersection_configs/2x2.json")
    print(calculate_state_space_size(intersection, 8, 1))
