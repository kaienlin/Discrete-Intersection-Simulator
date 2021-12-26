from typing import Tuple, Set, Dict, Iterable
import copy

class Intersection:
    def __init__(self) -> None:
        self.__conflict_zones: Set[str] = set()
        self.__transitions: Set[Tuple[str, str]] = set()
        self.__adjacency_list: Dict[str, Set[str]] = dict()
        self.__trajectories: Set[Tuple[str]] = set()
        self.__src_lanes: Dict[str, Set[str]] = dict()
        self.__dst_lanes: Dict[str, Set[str]] = dict()
        self.__cz_coordinates: Dict[str, Tuple[float, float]] = dict()

    def add_conflict_zone(self, cz_id: str) -> None:
        '''
        Add a conflict zone to the intersection.
        '''
        if cz_id in ["^", "$"]:
            raise Exception("[Intersection.add_conflict_zone] invalid cz_id")
        self.__conflict_zones.add(cz_id)
        self.__adjacency_list[cz_id] = set()

    def add_transition(self, from_cz_id: str, to_cz_id: str) -> None:
        '''
        Add a transition to the intersection.
        A transition is an ordered pair of CZs representing a valid move between the two CZs.
        '''
        if from_cz_id not in self.__conflict_zones:
            raise Exception(f"[Intersection.add_edge] vertex {from_cz_id} does not exist.")
        if to_cz_id not in self.__conflict_zones:
            raise Exception(f"[Intersection.add_edge] vertex {to_cz_id} does not exist.")
        self.__transitions.add((from_cz_id, to_cz_id))
        self.__adjacency_list[from_cz_id].add(to_cz_id)

    def add_trajectory(self, trajectory: Tuple[str]) -> None:
        '''
        Add a trajectory to the intersection.
        If an adjacent pair in the trajectory is not in the set of edges, add it.
        '''
        if not any([trajectory[0] in lane_cz for lane_cz in self.__src_lanes.values()]):
            raise Exception("[Intersection.add_trajectory] trajectory not belongs to any source lane")
        if not any([trajectory[-1] in lane_cz for lane_cz in self.__dst_lanes.values()]):
            raise Exception("[Intersection.add_trajectory] trajectory not belongs to any destination lane")
        self.__trajectories.add(trajectory)
        for i in range(len(trajectory) - 1):
            if (trajectory[i], trajectory[i+1]) not in self.__transitions:
                self.__transitions.add((trajectory[i], trajectory[i+1]))

    def add_src_lane(self, src_lane_id: str, associated_CZs: Iterable[str]) -> None:
        '''
        Add a source lane identifier and its associated CZs to the intersection.
        '''
        if src_lane_id in ["^", "$"]:
            raise Exception("[Intersection.add_src_lane] invalid src_lane_id")
        if src_lane_id in self.__src_lanes:
            raise Exception(f"[Intersection.add_src_lane] src_lane_id {src_lane_id} already exists.")
        self.__src_lanes[src_lane_id] = set(associated_CZs)

    def add_dst_lane(self, dst_lane_id: str, associated_CZs: Iterable[str]) -> None:
        '''
        Add a destination lane identifier and its associated CZs to the intersection.
        '''
        if dst_lane_id in ["^", "$"]:
            raise Exception("[Intersection.add_dst_lane] invalid dst_lane_id")
        if dst_lane_id in self.__dst_lanes:
            raise Exception(f"[Intersection.add_dst_lane] dst_lane_id {dst_lane_id} already exists.")
        self.__dst_lanes[dst_lane_id] = set(associated_CZs)

    def set_cz_coordinate(self, cz_id: str, x: float, y: float) -> None:
        '''
        Set the coordinate of a conflict zone.
        This information may be used in rendering of state.
        '''
        if cz_id not in self.__conflict_zones:
            raise Exception(f"[Intersection.set_cz_coordinate] cz_id {cz_id} not in conflict zones")
        self.__cz_coordinates[cz_id] = (x, y)

    def get_traj_dst_lane(self, traj: Tuple[str]) -> str:
        assert(traj in self.__trajectories)
        for dst_lane_id, czs in self.__dst_lanes.items():
            if traj[-1] in czs:
                return dst_lane_id
        assert(False)

    @property
    def conflict_zones(self) -> Set[str]:
        return copy.deepcopy(self.__conflict_zones)

    @property
    def transitions(self) -> Set[Tuple[str, str]]:
        return copy.deepcopy(self.__transitions)

    @property
    def adjacency_list(self) -> Dict[str, Set[str]]:
        return copy.deepcopy(self.__adjacency_list)

    @property
    def trajectories(self) -> Set[Tuple[str]]:
        return copy.deepcopy(self.__trajectories)

    @property
    def src_lanes(self) -> Dict[str, Set[str]]:
        return copy.deepcopy(self.__src_lanes)

    @property
    def dst_lanes(self) -> Dict[str, Set[str]]:
        return copy.deepcopy(self.__dst_lanes)

