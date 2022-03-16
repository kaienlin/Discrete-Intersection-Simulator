from __future__ import annotations

from typing import Dict, Any, Set, List, Tuple
import json
import fcntl
import os
import errno

import numpy as np

from simulation import Intersection


class DynamicQtable:
    def __init__(self, action_num: int, init_state_num: int = 1<<16):
        self.__arr = np.zeros((init_state_num, action_num))
    
    def __getitem__(self, items):
        if type(items) is int:
            return self.access_row(items)
        elif type(items) is tuple:
            assert all([type(i) is int for i in items])
            assert len(items) <= 2
            return self.access_row(items[0])[items[1]]
        else:
            raise Exception("[DynamicQtable] unsupported indices")

    def __setitem__(self, items, values):
        if type(items) is int:
            np.copyto(self.access_row(items), values)
        elif type(items) is tuple:
            assert all([type(i) is int for i in items])
            assert len(items) <= 2
            np.put(self.access_row(items[0]), items[1], values)
        else:
            raise Exception("[DynamicQtable] unsupported indices")

    def access_row(self, row_index: int):
        while row_index >= self.__arr.shape[0]:
            self.__arr = np.append(self.__arr, np.zeros((self.__arr.shape[0], self.__arr.shape[1])), axis=0)
        return self.__arr[row_index]

    def save(self, path):
        np.save(path, self.__arr)

    def load(self, path):
        self.__arr = np.load(path)   


class Digraph:
    def __init__(self):
        self.name_to_idx: Dict[Any, int] = {}
        self.idx_to_name: Dict[int, Any] = {}
        self.adj: List[Set] = []

    @property
    def vertices(self) -> List:
        return list(self.name_to_idx.keys())

    def get_neighbors(self, v: Any) -> List:
        v_idx = self.name_to_idx[v]
        return [self.idx_to_name[u_idx] for u_idx in self.adj[v_idx]]

    def print(self) -> None:
        for i, neighbors in enumerate(self.adj):
            print(self.idx_to_name[i], end=": ")
            for u in sorted(neighbors):
                print(self.idx_to_name[u], end=" ")
            print()

    def add_vertex(self, v: Any) -> int:
        idx = self.name_to_idx.get(v, None)
        if idx is None:
            idx = len(self.adj)
            self.name_to_idx[v] = idx
            self.idx_to_name[idx] = v
            self.adj.append(set())
        return idx
    
    def add_edge(self, src: Any, dst: Any) -> None:
        src_idx = self.add_vertex(src)
        dst_idx = self.add_vertex(dst)
        self.adj[src_idx].add(dst_idx)

    def remove_edge(self, src: Any, dst: Any) -> None:
        src_idx = self.name_to_idx.get(src, None)
        dst_idx = self.name_to_idx.get(dst, None)
        if src_idx is not None and dst_idx is not None:
            self.adj[src_idx].remove(dst_idx)

    def has_cycle(self) -> bool:
        color: List[int] = [0 for _ in self.adj]
        
        def dfs(v: int) -> bool:
            color[v] = 1
            for u in self.adj[v]:
                if color[u] == 0:
                    if dfs(u):
                        return True
                elif color[u] == 1:
                    return True
            color[v] = 2
            return False

        for v in range(len(self.adj)):
            if color[v] == 0:
                if dfs(v):
                    return True
        
        return False

    def get_scc_graph(self) -> Digraph:
        adj_rev: List[Set] = [set() for _ in self.adj]
        for src, neighbors in enumerate(self.adj):
            for u in neighbors:
                adj_rev[u].add(src)
        
        visited: List = [False for _ in self.adj]
        order: List = []
        def dfs1(v: int) -> None:
            visited[v] = True
            for u in self.adj[v]:
                if not visited[u]:
                    dfs1(u)
            order.append(v)
        
        for v in range(len(self.adj)):
            if not visited[v]:
                dfs1(v)
        
        visited = [False for _ in self.adj]
        def dfs2(v: int, component: List) -> None:
            visited[v] = True
            component.append(v)
            for u in adj_rev[v]:
                if not visited[u]:
                    dfs2(u, component)

        scc_graph = type(self)()
        home: Dict[Any, Tuple] = {}
        for v in order[::-1]:
            if not visited[v]:
                component_idx = []
                dfs2(v, component_idx)
                component = tuple(sorted([self.idx_to_name[i] for i in component_idx]))
                scc_graph.add_vertex(component)
                for member in component_idx:
                    home[member] = component
        
        for component in scc_graph.vertices:
            for member in component:
                for u in self.adj[self.name_to_idx[member]]:
                    if component != home[u]:
                        scc_graph.add_edge(component, home[u])
        
        return scc_graph


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


class FileLock:
    def __init__(self, path, mode="exclusive"):
        self.path = path
        try:
            tmp_fd = os.open(path, os.O_CREAT | os.O_EXCL)
            os.close(tmp_fd)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        self.fd = os.open(path, os.O_RDWR)
        self.locked = False
        
        assert mode in ("exclusive", "shared")
        self.mode = mode

    def acquire(self):
        if self.locked:
            return
        if self.mode == "exclusive":
            fcntl.flock(self.fd, fcntl.LOCK_EX)
        elif self.mode == "shared":
            fcntl.flock(self.fd, fcntl.LOCK_SH)
        else:
            raise
        self.locked = True

    def release(self):
        if not self.locked:
            return
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)
        self.locked = False
    
    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, traceback):
        self.release()

    def __del__(self):
        self.release()
