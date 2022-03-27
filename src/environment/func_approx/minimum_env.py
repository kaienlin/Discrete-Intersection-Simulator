from typing import Iterable, Tuple
from copy import deepcopy

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np

from simulation import Simulator, Vehicle, VehicleState
from environment.raw_state import RawStateSimulatorEnv


class MinimumEnv(py_environment.PyEnvironment):
    '''
    A environment for deep reinforcement learning which conforms to
    the specification of TensorFlow Agents. The state of this environment
    contains the minimum information for an intersection manager to make
    decision.
    '''
    def __init__(
        self,
        sim: Simulator,
        max_vehicle_num: int,
        deadlock_cost: int = int(1e9)
    ):
        self.sim: Simulator = sim
        self.max_vehicle_num: int = max_vehicle_num
        self.deadlock_cost: int = deadlock_cost
        self.is_snapshot = False

        self.raw_state_env: RawStateSimulatorEnv \
            = RawStateSimulatorEnv(sim, self.deadlock_cost)
        self.prev_included_vehicles: Iterable[Vehicle] = tuple()
        self.sorted_cz_ids: Tuple[str] = tuple(sorted(self.sim.intersection.conflict_zones))

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=max_vehicle_num, name="action")

        self.field_sizes = (max(len(traj) for traj in self.sim.intersection.trajectories), 1, 1)
        self.state_size = sum(self.field_sizes) * max_vehicle_num
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.state_size,), dtype=np.int32, minimum=0, name="observation")

        self._state = np.zeros(self.state_size, dtype=np.int32)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def render(self):
        self.raw_state_env.render()

    def decode_state(self, state):
        vehicles = []
        sz = sum(self.field_sizes)
        for i in range(self.max_vehicle_num):
            vec = state[i * sz: i * sz + sz]
            if vec[self.field_sizes[0]] == 0:
                break
            vehicle = ([], vec[self.field_sizes[0]] - 2, bool(vec[self.field_sizes[0] + 1]))
            for j in range(self.field_sizes[0]):
                if vec[j] == 0:
                    break
                vehicle[0].append(self.sorted_cz_ids[vec[j] - 1])
            vehicles.append(vehicle)
        return vehicles

    def _reset(self, new_sim=None):
        if self.is_snapshot:
            self.is_snapshot = False
            return ts.restart(self._state)
        self.raw_state_env.reset(new_sim=new_sim)
        _, vehicles, _ = self.raw_state_env.history[-1]
        self._state, self.prev_included_vehicles = self._encode_state_from_vehicles(vehicles)
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action: int):
        if self._episode_ended:
            return self.reset()

        acted_vehicle_id: str = ""
        raw_action: int = 0

        if 0 < action <= len(self.prev_included_vehicles) \
            and self.prev_included_vehicles[action - 1].state == VehicleState.WAITING:
            acted_vehicle_id = self.prev_included_vehicles[action - 1].id

            for i, vehicle in enumerate(self.raw_state_env.history[-1][1]):
                if vehicle.id == acted_vehicle_id:
                    raw_action = i + 1
                    break

        _, delayed_time, self._episode_ended, _ = self.raw_state_env.step(raw_action)

        cur_vehicles: Iterable[Vehicle] = self.raw_state_env.history[-1][1]
        next_state, self.prev_included_vehicles = self._encode_state_from_vehicles(cur_vehicles)

        reward = -delayed_time

        if self._episode_ended:
            return ts.termination(next_state, reward)

        return ts.transition(next_state, reward=reward, discount=1.0)

    def get_snapshots(self):
        res = []
        vehicle_ids_prev = set()
        for i, (t_0, raw_vehicles_0, _) in enumerate(self.raw_state_env.history[:-1]):
            S_0, vehicles_0 = self._encode_state_from_vehicles(raw_vehicles_0)
            sim = self.raw_state_env.sim_snapshots[i]

            vehicle_ids_0 = {vehicle.id for vehicle in vehicles_0}
            if vehicle_ids_0.issubset(vehicle_ids_prev):
                continue

            vehicle_ids_prev = vehicle_ids_0
            for vehicle in sim.vehicles:
                if vehicle.id not in vehicle_ids_0:
                    sim.remove_vehicle(vehicle.id)

            env_snapshot = type(self)(
                sim,
                max_vehicle_num=self.max_vehicle_num,
                deadlock_cost=self.deadlock_cost
            )

            env_snapshot.prev_included_vehicles = deepcopy(vehicles_0)
            env_snapshot.raw_state_env.history.append([t_0, deepcopy(vehicles_0), ""])
            env_snapshot._state = S_0
            env_snapshot.is_snapshot = True

            res.append((env_snapshot._state, env_snapshot))

        return res

    def _encode_state_from_vehicles(self, vehicles: Iterable[Vehicle]):
        vehicles_near_intersection = []
        for vehicle in vehicles:
            if vehicle.state not in (VehicleState.LEFT, VehicleState.NOT_ARRIVED):
                vehicles_near_intersection.append(vehicle)

        def vehicle_priority_func(vehicle: Vehicle) -> int:
            if 0 <= vehicle.idx_on_traj <= len(vehicle.trajectory) - 1:
                return -1
            if vehicle.state == VehicleState.WAITING:
                return -1
            if vehicle.idx_on_traj == -1 \
                and vehicle.state in [VehicleState.WAITING, VehicleState.BLOCKED]:
                num_pred_vehicles = 0
                for other in vehicles:
                    if other.src_lane_id == vehicle.src_lane_id \
                        and other.idx_on_traj == -1 \
                        and other.state in [VehicleState.WAITING, VehicleState.BLOCKED] \
                        and other.earliest_arrival_time < vehicle.earliest_arrival_time:
                        num_pred_vehicles += 1
                return num_pred_vehicles
            return 1000

        vehicles_near_intersection.sort(key=vehicle_priority_func)

        included_vehicles = []
        for vehicle in vehicles_near_intersection:
            included_vehicles.append(vehicle)
            if len(included_vehicles) == self.max_vehicle_num:
                break

        vehicle_state_list = []
        for vehicle in included_vehicles:
            vehicle_state = np.zeros(sum(self.field_sizes), dtype=np.int32)

            for i, cz_id in enumerate(vehicle.trajectory[max(vehicle.idx_on_traj, 0):]):
                vehicle_state[i] = self.sorted_cz_ids.index(cz_id) + 1

            vehicle_state[self.field_sizes[0]] = min(0, vehicle.idx_on_traj)

            if vehicle.state == VehicleState.WAITING:
                vehicle_state[self.field_sizes[0] + 1] = 1

            vehicle_state_list.append(vehicle_state)

        indices = sorted(range(len(vehicle_state_list)), key=lambda i: vehicle_state_list[i].sum())
        included_vehicles = [included_vehicles[i] for i in indices]

        while len(vehicle_state_list) < self.max_vehicle_num:
            vehicle_state_list.append(np.zeros(sum(self.field_sizes), dtype=np.int32))

        return np.concatenate(vehicle_state_list), included_vehicles
