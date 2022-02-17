import random
from collections import defaultdict

from environment.vehicle_based.SimulatorEnv import SimulatorEnv
from simulator.simulation import Simulator
from simulator.vehicle import VehicleState

class DisturbedSimulatorEnv(SimulatorEnv):
    def __init__(
        self,
        sim: Simulator,
        max_vehicle_num: int = 8,
        disturbation_prob: float = 0.01
    ):
        super().__init__(sim, max_vehicle_num)
        self.disturbance_prob: float = disturbation_prob

    def step(self, action: int):
        candidates = defaultdict(lambda: list())
        for vehicle in self.prev_vehicles:
            if vehicle.state == VehicleState.WAITING and random.uniform(0, 1) < self.disturbance_prob:
                next_cz = vehicle.get_next_cz()
                if next_cz == "$":
                    self.sim.simulation_step_act(vehicle.id)
                else:
                    candidates[next_cz].append(vehicle.id)

        for next_cz, candidate_vehicles in candidates.items():
            chosen = random.choice(candidate_vehicles)
            self.sim.simulation_step_act(chosen)

        return super().step(action)
