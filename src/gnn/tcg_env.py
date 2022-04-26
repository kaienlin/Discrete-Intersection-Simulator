from typing import Iterable

import numpy as np

from simulation import Intersection, Vehicle
from .tcg_numpy import TimingConflictGraphNumpy


class TcgEnv:
    def __init__(
        self,
        intersection: Intersection,
        vehicles: Iterable[Vehicle],
        ensure_deadlock_free: bool = True
    ):
        self.tcg: TimingConflictGraphNumpy \
            = TimingConflictGraphNumpy(intersection, vehicles)
        self.prev_delay_time: int = 0
        self.ensure_deadlock_free: bool = ensure_deadlock_free

    def done(self) -> bool:
        return self.tcg.is_scheduled.all()

    def make_state(self):
        adj: np.ndarray = self.tcg.t1_edge + self.tcg.t2_edge + self.tcg.t3_edge + self.tcg.t4_edge

        entering_time_lb: np.ndarray = self.tcg.entering_time_lb
        is_scheduled: np.ndarray = self.tcg.is_scheduled
        feature: np.ndarray = np.concatenate([
            entering_time_lb.reshape(-1, 1),
            is_scheduled.reshape(-1, 1)
        ], axis=1)

        front_vertices: np.ndarray = self.tcg.front_unscheduled_vertices
        mask: np.ndarray = self.tcg.schedulable_mask

        if self.ensure_deadlock_free:
            for i, vertex in enumerate(front_vertices):
                if mask[i]:
                    deadlock: bool = self.tcg.test_deadlock(vertex)
                    if deadlock:
                        mask[i] = np.False_

        return adj, feature, front_vertices, mask

    def step(self, action: int):
        self.tcg.schedule_vertex(action)
        adj, feature, front_vertices, mask = self.make_state()
        cur_delay_time = self.tcg.get_delay_time()
        reward = (self.prev_delay_time - cur_delay_time) / self.tcg.num_vehicles
        self.prev_delay_time = cur_delay_time
        return adj, feature, reward, self.done(), front_vertices, mask

    def reset(self, intersection: Intersection, vehicles: Iterable[Vehicle]):
        self.tcg = TimingConflictGraphNumpy(intersection, vehicles)
        self.prev_delay_time = 0
        return self.make_state()
