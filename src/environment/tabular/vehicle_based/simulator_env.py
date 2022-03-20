from copy import deepcopy
from typing import Tuple, Iterable, Set

from environment.tabular.vehicle_based.base import VehicleBasedStateEnv
from environment.raw_state import RawStateSimulatorEnv
from simulation.simulator import Simulator
from simulation.vehicle import Vehicle, VehicleState


class SimulatorEnv(VehicleBasedStateEnv):
    def __init__(
        self,
        sim: Simulator,
        max_vehicle_num: int = 8,
        max_vehicle_num_per_src_lane: int = 1
    ):
        super().__init__(sim.intersection, max_vehicle_num)
        self.max_vehicle_num_per_src_lane: int = max_vehicle_num_per_src_lane
        
        self.raw_state_env: RawStateSimulatorEnv = RawStateSimulatorEnv(sim, self.deadlock_cost)

        self.prev_included_vehicles: Iterable[Vehicle] = tuple()

    @property
    def sim(self) -> Simulator:
        return self.raw_state_env.sim

    def reset(self, new_sim=None):
        self.raw_state_env.reset(new_sim=new_sim)
        _, vehicles, _ = self.raw_state_env.history[-1]
        state, self.prev_included_vehicles = self._encode_state_from_vehicles(vehicles)
        return state

    def render(self) -> None:
        print(f"State at time {self.raw_state_env.history[-1][0]}")
        for vehicle in self.prev_included_vehicles:
            print(f"{vehicle.id}: ", end="")
            print(f"({vehicle.get_cur_cz()}) -> ({vehicle.get_next_cz()})", end="")
            print(f"; [{vehicle.state.name.lower()}]", end="")
            print("")

    def step(self, action: int) -> Tuple:
        assert 0 <= action <= len(self.prev_included_vehicles)
        acted_vehicle_id: str = "" if action == 0 else self.prev_included_vehicles[action - 1].id
        
        raw_action: int = 0
        if action > 0:
            prev_vehicles: Iterable[Vehicle] = self.raw_state_env.history[-1][1]
            raw_action = 1
            while raw_action <= len(prev_vehicles):
                if prev_vehicles[raw_action - 1].id == acted_vehicle_id:
                    break
                raw_action += 1
            else:
                assert False

        _, delayed_time, terminal, _ = self.raw_state_env.step(raw_action)

        cur_vehicles: Iterable[Vehicle] = self.raw_state_env.history[-1][1]
        next_state, self.prev_included_vehicles = self._encode_state_from_vehicles(cur_vehicles)

        return next_state, delayed_time, terminal, {}

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
                max_vehicle_num_per_src_lane=self.max_vehicle_num_per_src_lane
            )

            env_snapshot.prev_included_vehicles = deepcopy(vehicles_0)
            env_snapshot.decoding_table = self.decoding_table
            env_snapshot.encoding_table = self.encoding_table
            env_snapshot.raw_state_env.history.append([t_0, deepcopy(vehicles_0), ""])

            res.append((S_0, env_snapshot))

        return res

    def _encode_state_from_vehicles(self, vehicles: Iterable[Vehicle]) -> Tuple:
        vehicles_near_intersection = []
        for vehicle in vehicles:
            if vehicle.state != VehicleState.LEFT \
                and vehicle.state != VehicleState.NOT_ARRIVED:
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
        lane_quota = {src_lane_id: self.max_vehicle_num_per_src_lane
                      for src_lane_id in self.sim.intersection.src_lanes}

        included_vehicles = []
        for vehicle in vehicles_near_intersection:
            if vehicle.get_cur_cz() == "^":
                if lane_quota[vehicle.src_lane_id] > 0:
                    lane_quota[vehicle.src_lane_id] -= 1
                    included_vehicles.append(vehicle)
            else:
                included_vehicles.append(vehicle)
            if len(included_vehicles) == self.max_vehicle_num:
                break

        res = []
        for vehicle in included_vehicles:
            res.append(self.VehicleState(
                src_lane="",
                trajectory=vehicle.trajectory[max(0, vehicle.idx_on_traj):],
                position=min(0, vehicle.idx_on_traj),
                state="waiting" if vehicle.state == VehicleState.WAITING else "non-waiting"
            ))
        res = tuple(res)
        indices = sorted(list(range(len(res))), key=lambda i: res[i])
        included_vehicles = [included_vehicles[i] for i in indices]
        return self.encode_state(res), included_vehicles
