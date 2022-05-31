from typing import Iterable

import numpy as np

from simulation import Intersection, Vehicle
from cycle_finder import CycleFinder
from .tcg_numpy import TimingConflictGraphNumpy


class TcgEnv:
    def __init__(
        self,
        ensure_deadlock_free: bool = True
    ):
        self.ensure_deadlock_free: bool = ensure_deadlock_free

    def done(self) -> bool:
        return self.tcg.is_scheduled.all()

    def _test_deadlock(self, vertex: int) -> bool:
        vertex2 = self.tcg.t1_next[vertex]
        if vertex2 == -1:
            return False
        cz2 = self.tcg.vertex_to_vehicle_cz[vertex2][1]
        cz1 = self.tcg.vertex_to_vehicle_cz[vertex][1]       

        vertex0 = self.tcg.t1_prev[vertex]
        if vertex0 != -1:
            cz0 = self.tcg.vertex_to_vehicle_cz[vertex0][1]
            self.transitions.remove((cz0, cz1))

        self.transitions.add((cz1, cz2))
        deadlock = self.cycle_finder.has_cycle(self.transitions, (cz1, cz2))
        self.transitions.remove((cz1, cz2))
        if vertex0 != -1:
            self.transitions.add((cz0, cz1))

        return deadlock

    def make_state(self):
        adj: np.ndarray = self.tcg.t1_edge + self.tcg.t3_edge_min + self.tcg.t4_edge_min
        adj = (adj != 0).astype(np.single)
        adj += np.eye(adj.shape[0], dtype=np.single)

        entering_time_lb: np.ndarray = self.tcg.entering_time_lb
        is_scheduled: np.ndarray = self.tcg.is_scheduled
        feature: np.ndarray = np.concatenate([
            entering_time_lb.reshape(-1, 1).astype(np.single) / 10,
            is_scheduled.reshape(-1, 1).astype(np.single)
        ], axis=1)

        front_vertices: np.ndarray = self.tcg.front_unscheduled_vertices.astype(np.int64)
        mask: np.ndarray = self.tcg.schedulable_mask

        if self.ensure_deadlock_free:
            for i, vertex in enumerate(front_vertices):
                if not mask[i]:
                    deadlock: bool = self._test_deadlock(vertex)
                    if deadlock:
                        mask[i] = 1
        return adj, feature, front_vertices, mask

    def step(self, action: int):
        self.tcg.schedule_vertex(action)
        
        cz0 = self.tcg.vertex_to_vehicle_cz[self.tcg.t1_prev[action]][1]
        cz1 = self.tcg.vertex_to_vehicle_cz[action][1]
        cz2 = self.tcg.vertex_to_vehicle_cz[self.tcg.t1_next[action]][1]
        if self.tcg.t1_prev[action] != -1:
            self.transitions.remove((cz0, cz1))
        if self.tcg.t1_next[action] != -1:
            self.transitions.add((cz1, cz2))

        adj, feature, front_vertices, mask = self.make_state()
        cur_delay_time = self.tcg.get_delay_time()
        reward = (self.prev_delay_time - cur_delay_time)
        self.prev_delay_time = cur_delay_time
        return adj, feature, reward, self.done(), front_vertices, mask

    def reset(self, intersection: Intersection, vehicles: Iterable[Vehicle]):
        self.tcg = TimingConflictGraphNumpy(intersection, vehicles)
        self.prev_delay_time = 0
        self.cycle_finder = CycleFinder(intersection)
        self.transitions = set()
        return self.make_state()
