import copy, sys
from typing import Tuple, Iterable, Set

from environment.vehicle_based.BaseEnv import VehicleBasedStateEnv
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
        self.sim: Simulator = sim
        self.max_vehicle_num_per_src_lane: int = max_vehicle_num_per_src_lane

        self.prev_vehicles: Tuple[Vehicle] = tuple()
        self.prev_timestamp: int = 0
        self.prev_idle_vehicles: Set[str] = set()

    def reset(self, new_sim=None):
        if new_sim is not None:
            self.sim = new_sim

        self.sim.reset_simulation()
        self.sim.run()

        timestamp, vehicles = self.sim.simulation_step_report()
        self.prev_timestamp = timestamp
        self.prev_state, self.prev_vehicles = self._encode_state_from_vehicles(vehicles)
        self.prev_idle_veh = set([veh.id for veh in vehicles if self._is_idle_state(veh.state)])

        return self.prev_state

    def render(self) -> None:
        print(f"State at time {self.prev_timestamp}")
        for veh in self.prev_vehicles:
            print(f"  - {veh.id} @{veh.get_cur_cz()} :{veh.state}")

    def step(self, action: int):
        assert 0 <= action <= len(self.prev_vehicles)
        veh_id: str = "" if action == 0 else self.prev_vehicles[action - 1].id
        self.sim.simulation_step_act(veh_id)

        waiting_time_sum = 0

        timestamp, vehicles = self.sim.simulation_step_report()
        num_waiting = len(self.prev_idle_veh) - (1 if veh_id in self.prev_idle_veh else 0)
        waiting_time_sum += (timestamp - self.prev_timestamp) * num_waiting
        self.prev_timestamp = timestamp

        next_state, included_vehicles = self._encode_state_from_vehicles(vehicles)
        self.prev_state = next_state
        self.prev_vehicles = included_vehicles
        self.prev_idle_veh = set([veh.id for veh in vehicles if self._is_idle_state(veh.state)])

        terminal = self.sim.status != "RUNNING"
        if terminal and self.sim.status == "DEADLOCK":
            waiting_time_sum += self.DEADLOCK_COST

        return next_state, waiting_time_sum, terminal, {}

    def _is_idle_state(self, state: VehicleState) -> bool:
        return state == VehicleState.WAITING or state == VehicleState.BLOCKED

    def _encode_state_from_vehicles(self, vehicles: Iterable[Vehicle]) -> Tuple:
        vehicles_near_intersection = []
        for vehicle in vehicles:
            if vehicle.state != VehicleState.LEFT and \
                not (vehicle.idx_on_traj == -1 and vehicle.state not in [VehicleState.WAITING, VehicleState.BLOCKED]):
                vehicles_near_intersection.append(vehicle)

        def vehicle_priority_func(vehicle: Vehicle) -> int:
            if 0 <= vehicle.idx_on_traj <= len(vehicle.trajectory) - 1:
                return -1
            if vehicle.state == VehicleState.WAITING:
                return -1
            if vehicle.idx_on_traj == -1 and vehicle.state in [VehicleState.WAITING, VehicleState.BLOCKED]:
                num_pred_vehicles = 0
                for other in vehicles:
                    if other.src_lane_id == vehicle.src_lane_id \
                        and other.idx_on_traj == -1 and other.state in [VehicleState.WAITING, VehicleState.BLOCKED] \
                        and other.earliest_arrival_time < vehicle.earliest_arrival_time:
                        num_pred_vehicles += 1
                return num_pred_vehicles

        vehicles_near_intersection.sort(key=vehicle_priority_func)
        lane_quota = {src_lane_id: self.max_vehicle_num_per_src_lane for src_lane_id in self.sim.intersection.src_lanes}
        
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
