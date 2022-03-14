from typing import Tuple, Optional, Iterable, Set, List
from copy import deepcopy

from simulation import Intersection, Simulator, VehicleState, Vehicle


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
        deadlock_cost: int = 1e9
    ):
        self.sim: Simulator = simulator
        self.intersection: Intersection = simulator.intersection
        self.deadlock_cost: int = deadlock_cost

        self.history: List[List[int, Iterable[Vehicle], str]] = []
        self.sim_snapshots: List[Simulator] = []

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

        self.sim.reset_simulation()
        self.sim.run()

        timestamp, vehicles = self.sim.simulation_step_report()
        self.history = [[timestamp, deepcopy(vehicles), ""]]
        self.sim_snapshots = [deepcopy(self.sim)]

        return vehicles

    def step(self, action: int) -> Tuple:
        prev_timestamp, prev_vehicles, _ = self.history[-1]

        if action < 0 or action > len(prev_vehicles) \
            or (action > 0 and prev_vehicles[action-1].state != VehicleState.WAITING):
            raise Exception(f"[RawStateSimulatorEnv] Invalid action {action}")

        acted_vehicle_id: str = "" if action == 0 else prev_vehicles[action-1].id
        self.sim.simulation_step_act(acted_vehicle_id)

        cur_timestamp, cur_vehicles = self.sim.simulation_step_report()
        self.history[-1][2] = acted_vehicle_id
        self.history.append([cur_timestamp, deepcopy(cur_vehicles), ""])
        self.sim_snapshots.append(deepcopy(self.sim))

        # Calculate the delayed time cumulated from
        # the previous times step to the current time step
        prev_idle_ids: Set = {vehicle.id for vehicle in prev_vehicles
                            if self.is_idle_state(vehicle.state)}
        if acted_vehicle_id != "":
            prev_idle_ids.remove(acted_vehicle_id)
        delayed_time: int = (cur_timestamp - prev_timestamp) * len(prev_idle_ids)

        terminal = self.sim.status != "RUNNING"
        if terminal and self.sim.status == "DEADLOCK":
            delayed_time += self.deadlock_cost

        return deepcopy(cur_vehicles), delayed_time, terminal, {}

    @staticmethod
    def is_idle_state(vehicle_state: VehicleState) -> bool:
        return vehicle_state in (VehicleState.BLOCKED, VehicleState.WAITING)
