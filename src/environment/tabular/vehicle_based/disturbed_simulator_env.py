import random

from environment.tabular.vehicle_based.simulator_env import SimulatorEnv
from simulation.simulator import Simulator
from simulation.vehicle import VehicleState


class DisturbedSimulatorEnv(SimulatorEnv):
    def __init__(
        self,
        sim: Simulator,
        max_vehicle_num: int = 8,
        disturbance_prob: float = 0.05
    ):
        super().__init__(sim, max_vehicle_num)
        self.disturbance_prob: float = disturbance_prob

    def step(self, action: int):
        assert 0 <= action <= len(self.prev_vehicles)

        if random.uniform(0, 1) < self.disturbance_prob:
            candidates = []
            for i, vehicle in enumerate(self.prev_vehicles):
                if i != action - 1 and vehicle.state == VehicleState.WAITING:
                    candidates.append(vehicle.id)

            if len(candidates) > 0:
                chosen: str = random.choice(candidates)
                self.sim.simulation_step_act(chosen)

        veh_id: str = "" if action == 0 else self.prev_vehicles[action - 1].id
        if veh_id == "" or self.prev_vehicles[action - 1].state == VehicleState.WAITING:
            self.sim.simulation_step_act(veh_id)

        timestamp, vehicles = self.sim.simulation_step_report()
        num_waiting = len(self.prev_idle_veh) - (1 if veh_id in self.prev_idle_veh else 0)
        waiting_time_sum = (timestamp - self.prev_timestamp) * num_waiting
        self.prev_timestamp = timestamp

        next_state, included_vehicles = self._encode_state_from_vehicles(vehicles)
        self.prev_state = next_state
        self.prev_vehicles = included_vehicles
        self.prev_idle_veh = {veh.id for veh in vehicles if self._is_idle_state(veh.state)}

        terminal = self.sim.status != "RUNNING"
        if terminal and self.sim.status == "DEADLOCK":
            waiting_time_sum += self.DEADLOCK_COST

        return next_state, waiting_time_sum, terminal, {}
