import pprint
import gym
from gym import spaces
from typing import Tuple, Dict, List, Set
from functools import lru_cache
import copy
import math
import sys

from simulator.intersection import Intersection

TRAFFIC_DENSITY = 0.05

class ProbabilisticEnv(gym.Env):
    TERMINAL_STATE = 0
    DEADLOCK_COST = 1e9
    def __init__(self, intersection: Intersection, queue_size_scale: Tuple[int] = (1,)):
        super(ProbabilisticEnv, self).__init__()
        self.intersection = intersection
        self.queue_size_scale = queue_size_scale

        self.sorted_src_lane_ids = sorted(intersection.src_lanes.keys())
        self.sorted_cz_ids = sorted(intersection.conflict_zones)

        self.state_space_size = self.__create_state_encoding(intersection)
        self.action_space_size = self.__create_action_encoding(intersection)

        self.deadlock_state_table = [False for _ in range(self.state_space_size)]
        for s in range(self.state_space_size):
            if self.__is_deadlock_state(self.outer_to_real_state[s]):
                self.deadlock_state_table[s] = True

        self.P = tuple([
            tuple([self.get_transitions(s, a) for a in range(self.action_space_size)])
            for s in range(self.state_space_size
        )])

        self.observation_space = spaces.Discrete(self.state_space_size)
        self.action_space = spaces.Discrete(self.action_space_size)

    def __calc_queue_size_scale(self, queue_size: int):
        for idx, scale in enumerate(self.queue_size_scale):
            if queue_size < scale:
                return idx
        return len(self.queue_size_scale)

    def __scale_to_real_size(self, scale: int):
        if scale == 0:
            return 0
        return self.queue_size_scale[scale - 1]

    def __create_action_encoding(self, intersection: Intersection) -> int:
        return len(intersection.conflict_zones) + len(intersection.src_lanes) + 1

    def __create_state_encoding(self, intersection: Intersection) -> int:
        trans_per_cz_id: Dict[str, Set[str]] = {cz_id: set() for cz_id in intersection.conflict_zones}
        for traj in sorted(intersection.trajectories):
            for idx, cz_1 in enumerate(traj[:-1]):
                cz_2 = traj[idx + 1]
                trans_per_cz_id[cz_1].add(cz_2)
            trans_per_cz_id[traj[-1]].add("$")
        self.trans_per_cz_id: Dict[str, List[str]] = {k: sorted(list(s)) for k, s in trans_per_cz_id.items()}
        self.trans_per_src_lane: Dict[str, List[str]] = {k: sorted(list(s)) for k, s in self.intersection.src_lanes.items()}

        # number of states of vehicle position
        n_states = 1
        for _, associated_czs in sorted(intersection.src_lanes.items()):
            n_states *= 2 * (len(associated_czs) + 1)
        for _, cz_ids in sorted(trans_per_cz_id.items()):
            n_states *= len(cz_ids) + 1
        
        # number of (not left) vehicles in each source lane
        n_states *= (len(self.queue_size_scale) + 1) ** len(self.sorted_src_lane_ids)

        n_outer_states = 0
        outer_to_real_state = dict()
        real_to_outer_state = dict()
        for s in range(n_states):
            if not self.__is_invalid_state(s):
                outer_to_real_state[n_outer_states] = s
                real_to_outer_state[s] = n_outer_states
                n_outer_states += 1
        self.outer_to_real_state: Dict[int, int] = outer_to_real_state
        self.real_to_outer_state: Dict[int, int] = real_to_outer_state

        return n_outer_states

    def is_invalid_state(self, state: int) -> bool:
        state = self.outer_to_real_state[state]
        return self.__is_invalid_state(state)

    def __is_invalid_state(self, real_state: int) -> bool:
        state_dict = self.__decode_state(real_state)
        occupied_cz = set()
        for cz_id, info in state_dict["cz_state"].items():
            if "next_pos" in info:
                occupied_cz.add(cz_id)
        for _, info in state_dict["src_lane_state"].items():
            if info.get("waiting", False) and info["next_pos"] in occupied_cz:
                return True
        for _, info in state_dict["cz_state"].items():
            if info.get("waiting", False) and info["next_pos"] in occupied_cz:
                return True
        return False

    def __is_deadlock_state(self, real_state: int) -> bool:
        state_dict = self.__decode_state(real_state)
        adj = {cz_id: [] for cz_id in self.sorted_cz_ids}
        for cz_id, info in state_dict["cz_state"].items():
            next_pos = info.get("next_pos", "")
            if next_pos and next_pos != "$":
                adj[cz_id].append(next_pos)
        
        color = {cz_id: "w" for cz_id in self.sorted_cz_ids}
        def dfs(v: str):
            color[v] = "g"
            for u in adj[v]:
                if color[u] == "w":
                    if dfs(u):
                        return True
                elif color[u] == "g":
                    return True
            color[v] = "b"
            return False
        for cz_id in self.sorted_cz_ids:
            if color[cz_id] == "w":
                if dfs(cz_id):
                    return True
        return False

    def encode_action(self, action: Dict) -> int:
        if "type" not in action:
            return 0
        num_src_lane = len(self.sorted_src_lane_ids)
        num_cz = len(self.sorted_src_lane_ids)
        if action["type"] == "src":
            return 1 + self.sorted_src_lane_ids.index(action["id"])
        if action["type"] == "cz":
            return 1 + num_src_lane + self.sorted_cz_ids.index(action["id"])

    @lru_cache(maxsize=128)
    def decode_action(self, action: int) -> Dict:
        if action == 0:
            return {}
        num_src_lane = len(self.sorted_src_lane_ids)
        num_cz = len(self.sorted_src_lane_ids)
        if 1 <= action <= num_src_lane:
            return {"type": "src", "id": self.sorted_src_lane_ids[action - 1]}
        elif num_src_lane + 1 <= action <= num_cz + num_src_lane:
            return {"type": "cz", "id": self.sorted_cz_ids[action - num_src_lane - 1]}

    def encode_state(self, state_dict: Dict) -> int:
        state = 0
        for cz_id in self.sorted_cz_ids:
            trans = self.trans_per_cz_id[cz_id]
            state *= 2 * (len(trans) + 1)
            next_pos = state_dict["cz_state"][cz_id].get("next_pos", False)
            is_waiting = state_dict["cz_state"][cz_id].get("waiting", False)
            if next_pos:
                state += 2 * (trans.index(next_pos) + 1)
            if is_waiting:
                state += 1
        
        for src_lane_id in self.sorted_src_lane_ids:
            trans = self.trans_per_src_lane[src_lane_id]
            state *= len(trans) + 1
            next_pos = state_dict["src_lane_state"][src_lane_id].get("next_pos", False)
            if next_pos:
                state += trans.index(next_pos) + 1
        
        for src_lane_id in self.sorted_src_lane_ids:
            queue_size = state_dict["src_lane_state"][src_lane_id]["queue_size"]
            scale = self.__calc_queue_size_scale(queue_size)
            state *= len(self.queue_size_scale) + 1
            state += scale
        
        try:
            return self.real_to_outer_state[state]
        except KeyError:
            pprint.pprint(self.__decode_state(state))

    @lru_cache(maxsize=1024)
    def __decode_state(self, real_state: int) -> Dict:
        res: Dict = {
            "src_lane_state": {src_lane_id: {} for src_lane_id in self.sorted_src_lane_ids},
            "cz_state": {cz_id: {} for cz_id in self.sorted_cz_ids}
        }

        for src_lane_id in self.sorted_src_lane_ids[::-1]:
            scale = real_state % (len(self.queue_size_scale) + 1)
            real_state //= (len(self.queue_size_scale) + 1)
            queue_size = 0 if scale == 0 else self.queue_size_scale[scale-1]
            res["src_lane_state"][src_lane_id]["queue_size"] = queue_size

        for src_lane_id in self.sorted_src_lane_ids[::-1]:
            trans = self.trans_per_src_lane[src_lane_id]
            lane_state = real_state % (len(trans) + 1)
            real_state //= len(trans) + 1
            if lane_state > 0:
                res["src_lane_state"][src_lane_id].update({
                        "waiting": True,
                        "next_pos": trans[lane_state - 1]
                })
            
        for cz_id in self.sorted_cz_ids[::-1]:
            trans = self.trans_per_cz_id[cz_id]
            cz_state = real_state % (2 * (len(trans) + 1))
            real_state //= 2 * (len(trans) + 1)
            is_waiting = bool(cz_state % 2)
            cz_state //= 2
            if cz_state > 0:
                res["cz_state"][cz_id].update({
                    "waiting": is_waiting,
                    "next_pos": trans[cz_state - 1]
                })
            
        return res

    def decode_state(self, state: int) -> Dict:
        real_state = self.outer_to_real_state[state]
        return self.__decode_state(real_state)

    def get_transitions(self, s: int, a: int):
        '''
        return a list of 3-tuple (prob, next_state, cost)
        '''
        if s == self.TERMINAL_STATE:
            return [(1.0, self.TERMINAL_STATE, 0)]
        elif self.deadlock_state_table[s]:
            return [(1.0, self.TERMINAL_STATE, self.DEADLOCK_COST)]
        res: List[Tuple[float, int, int]] = []
        s_dec = self.decode_state(s)
        a_dec = self.decode_action(a)

        cost = 0
        for info in s_dec["src_lane_state"].values():
            next_pos = info.get("next_pos", "")
            if next_pos:
                cost += 1
            cost += info["queue_size"]
        for info in s_dec["cz_state"].values():
            next_pos = info.get("next_pos", "")
            if next_pos:
                cost += 1

        def explore_cz(i: int, prob: float, sp: dict):
            if i == len(self.sorted_cz_ids):
                #print("* ", prob)
                #pprint.pprint(sp)
                res.append((prob, self.encode_state(sp), cost))
                return
            # If the cz is just occupied by a the vehicle specified in action,
            # then it is impossible for the vehicle to be waiting suddenly
            cz_id = self.sorted_cz_ids[i]
            next_pos = s_dec["cz_state"][cz_id].get("next_pos", False)
            is_waiting = s_dec["cz_state"][cz_id].get("waiting", False)
            if next_pos and not is_waiting and (next_pos == "$" or not sp["cz_state"][next_pos].get("next_pos", False)):
                waiting_prob = 0.067
                sp["cz_state"][cz_id]["waiting"] = True
                #print(f"cz {cz_id} waiting")
                explore_cz(i + 1, prob * waiting_prob, sp)
                sp["cz_state"][cz_id]["waiting"] = False
                #print(f"cz {cz_id} keep")
                explore_cz(i + 1, prob * (1 - waiting_prob), sp)
            else:
                #print(f"cz {cz_id} cannot change")
                explore_cz(i + 1, prob, sp)

        def explore_queue(i: int, prob: float, sp: dict):
            # a vehicle arrived
            if i == len(self.sorted_src_lane_ids):
                explore_cz(0, prob, sp) ##
                return
            src_lane_id = self.sorted_src_lane_ids[i]
            cur_scale_size = sp["src_lane_state"][src_lane_id]["queue_size"]
            cur_scale = self.__calc_queue_size_scale(cur_scale_size)
            if cur_scale < len(self.queue_size_scale):
                next_scale_size = self.queue_size_scale[cur_scale]
                increase_prob = TRAFFIC_DENSITY * 1.0 / (next_scale_size - cur_scale_size)
                unchange_prob = 1.0 - increase_prob
                # queue size increases
                if not math.isclose(0.0, unchange_prob):
                    #print(f"queue {src_lane_id} unchange")
                    explore_src(i, prob * unchange_prob, sp) ##

                # queue size not increases
                if not math.isclose(0.0, increase_prob):
                    sp["src_lane_state"][src_lane_id]["queue_size"] = next_scale_size
                    #print(f"queue {src_lane_id} increases")
                    explore_src(i, prob * increase_prob, sp)  ##
                    sp["src_lane_state"][src_lane_id]["queue_size"] = cur_scale_size
            else:
                explore_src(i, prob, sp) ##

        def explore_src(i: int, prob: float, sp: Dict):
            if i == len(self.sorted_src_lane_ids):
                explore_cz(0, prob, sp)
                return
            src_lane_id = self.sorted_src_lane_ids[i]
            if not sp["src_lane_state"][src_lane_id].get("waiting", False):
                # change to waiting if a vehicle in queue arrived
                if sp["src_lane_state"][src_lane_id]["queue_size"] > 0:
                    trans = self.trans_per_src_lane[src_lane_id]
                    avail_trans_cnt = 0
                    for next_cz in trans:
                        if sp["cz_state"][next_cz].get("next_pos", "") == "":
                            avail_trans_cnt += 1
                            sp["src_lane_state"][src_lane_id]["next_pos"] = next_cz
                            sp["src_lane_state"][src_lane_id]["waiting"] = True

                            # calculate the probability that the queue size decreases for one level
                            cur_queue_size = sp["src_lane_state"][src_lane_id]["queue_size"]
                            cur_queue_scale = self.__calc_queue_size_scale(cur_queue_size)
                            next_queue_size = 0 if cur_queue_scale == 1 else self.queue_size_scale[cur_queue_scale - 2]
                            decrease_prob = 1.0 / (cur_queue_size - next_queue_size)

                            # Case 1: queue size does not decreases
                            if not math.isclose(0.0, 1 - decrease_prob):
                                #print(f"src {src_lane_id} not decrease and wait for {next_cz}")
                                explore_src(i + 1, prob * (1 - decrease_prob) * (1 / len(trans)), sp) ##
                            
                            # Case 2: queue size decreases
                            if not math.isclose(0.0, decrease_prob):
                                sp["src_lane_state"][src_lane_id]["queue_size"] = next_queue_size
                                #print(f"src {src_lane_id} decreases and wait for {next_cz}")
                                explore_src(i + 1, prob * decrease_prob * (1 / len(trans)), sp) ##
                                sp["src_lane_state"][src_lane_id]["queue_size"] = cur_queue_size

                            del sp["src_lane_state"][src_lane_id]["next_pos"]
                            del sp["src_lane_state"][src_lane_id]["waiting"]
                    
                    # For the blocked transitions
                    #print(f"src {src_lane_id} unchanged")
                    if avail_trans_cnt < len(trans):
                        explore_src(i + 1, prob * (1.0 - avail_trans_cnt / len(trans)), sp) ##
                else:
                    # queue size = 0
                    #print(f"src {src_lane_id} unchanged")
                    explore_src(i + 1, prob, sp) ##
            else:
                explore_src(i + 1, prob, sp)


        action_type: str = ""
        if a_dec.get("type", "") == "src":
            if s_dec["src_lane_state"][a_dec["id"]].get("waiting", False):
                action_type = "src_lane_state"
        elif a_dec.get("type", "") == "cz":
            if s_dec["cz_state"][a_dec["id"]].get("waiting", False):
                action_type = "cz_state"
        
        state_dict = copy.deepcopy(s_dec)

        if action_type != "" and s_dec[action_type][a_dec["id"]].get("waiting", False):
            next_pos = s_dec[action_type][a_dec["id"]]["next_pos"]
            del state_dict[action_type][a_dec["id"]]["waiting"]
            del state_dict[action_type][a_dec["id"]]["next_pos"]

            for _, info in state_dict["src_lane_state"].items():
                if info.get("waiting", False):
                    cost = 0
            for _, info in state_dict["cz_state"].items():
                if info.get("waiting", False):
                    cost = 0

            if next_pos == "$":
                explore_src(0, 1.0, state_dict)
            else:
                trans = self.trans_per_cz_id[next_pos]
                for next2_pos in trans:
                    state_dict["cz_state"][next_pos].update({
                        "waiting": False,
                        "next_pos": next2_pos
                    })
                    for _, info in state_dict["src_lane_state"].items():
                        if info.get("waiting", False) and info.get("next_pos", "") == next_pos:
                            del info["waiting"]
                            del info["next_pos"]
                            if info["queue_size"] == 0:
                                info["queue_size"] += 1

                    for _, info in state_dict["cz_state"].items():
                        if info.get("waiting", False) and info.get("next_pos", "") == next_pos:
                            info["waiting"] = False 
                    explore_src(0, 1.0 / len(trans), state_dict)
        else:
            explore_src(0, 1.0, state_dict)

        return res

