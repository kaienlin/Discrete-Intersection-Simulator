from typing import Tuple, Optional, Iterable, Set, List
from copy import deepcopy

from simulation import Intersection, Simulator, SimulatorStatus, VehicleState, Vehicle


class RawStateSimulatorEnv:
    '''
    This the reinforcement learning environment which
    return the raw vehicle objects as states. One can
    further simplify or organize the information in
    the vehicle objects for their own state definition.
    '''
    def __init__(
        self,
        simulator: Simulator,
        deadlock_cost: int = int(1e9),
        snapshot: bool = True
    ):
        self.sim: Simulator = simulator
        self.intersection: Intersection = simulator.intersection
        self.deadlock_cost: int = deadlock_cost
        self.snapshot: bool = snapshot

        self.history: List[List[int, Iterable[Vehicle], str]] = []
        self.sim_snapshots: List[Simulator] = []
        self.prev_cumulative_delayed_time: int = 0

    def render(self) -> None:
        if len(self.history) == 0:
            return
        cur_timestamp, cur_vehicles, _ = self.history[-1]

        print(f"t = {cur_timestamp}")
        for vehicle in cur_vehicles:
            print(f"{vehicle.id}: ", end="")
            print(f"({vehicle.get_cur_cz()}) -> ({vehicle.get_next_cz()})", end="")
            print(f"; {vehicle.state.name.lower()}", end="")
            print("")

    def reset(self, new_sim: Optional[Simulator] = None):
        if new_sim is not None:
            self.sim = new_sim

        self.sim.start()

        observation = self.sim.observe()
        timestamp, vehicles = observation["time"], observation["vehicles"]
        self.history = [[timestamp, deepcopy(vehicles), ""]]
        if self.snapshot:
            self.sim_snapshots = [deepcopy(self.sim)]
        self.prev_cumulative_delayed_time = 0

        return vehicles

    def step(self, action: int) -> Tuple:
        prev_timestamp, prev_vehicles, _ = self.history[-1]

        if action < 0 or action > len(prev_vehicles) \
            or (action > 0 and prev_vehicles[action-1].state != VehicleState.READY):
            raise Exception(f"[RawStateSimulatorEnv] Invalid action {action}")

        acted_vehicle_id: str = None if action == 0 else prev_vehicles[action-1].id
        self.sim.step(acted_vehicle_id)

        observation = self.sim.observe()
        cur_timestamp, cur_vehicles = observation["time"], observation["vehicles"]
        self.history[-1][2] = acted_vehicle_id
        self.history.append([cur_timestamp, deepcopy(cur_vehicles), ""])
        if self.snapshot:
            self.sim_snapshots.append(deepcopy(self.sim))

        cur_cumulative_delayed_time: int = self.sim.get_cumulative_delayed_time()
        delayed_time = cur_cumulative_delayed_time - self.prev_cumulative_delayed_time
        self.prev_cumulative_delayed_time = cur_cumulative_delayed_time

        terminal: bool = self.sim.status != SimulatorStatus.RUNNING
        deadlock: bool = terminal and self.sim.status == SimulatorStatus.DEADLOCK
        if deadlock:
            delayed_time += self.deadlock_cost

        return deepcopy(cur_vehicles), delayed_time, terminal, {"deadlock": deadlock}

    @staticmethod
    def is_idle_state(vehicle_state: VehicleState) -> bool:
        return vehicle_state in (VehicleState.BLOCKED, VehicleState.READY)
