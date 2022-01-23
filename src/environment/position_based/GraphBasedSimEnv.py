import copy
from typing import Iterable, Set, Tuple

from simulator.simulation import Simulator
from simulator.vehicle import Vehicle, VehicleState
from environment.position_based.BaseEnv import PositionBasedStateEnv
 
class GraphBasedSimEnv(PositionBasedStateEnv):
    def __init__(self, sim: Simulator, queue_size_scale: Tuple[int] = (1,)):
        super().__init__(
            sim.intersection,
            queue_size_scale=queue_size_scale
        )
        self.sim: Simulator = sim

        # simulation-related objects for calculating cost
        self.prev_timestamp: int = 0
        self.prev_vehicles = None
        self.prev_idle_veh: Set[str] = set()

    def reset(self, new_sim=None):
        '''
        reset the environment and return a initial state
        '''
        if new_sim is not None:
            self.sim = new_sim

        self.sim.reset_simulation()
        self.sim.run()

        timestamp, vehicles = self.sim.simulation_step_report()
        self.prev_timestamp = timestamp
        self.prev_vehicles = copy.deepcopy(vehicles)
        self.prev_idle_veh = set([veh.id for veh in vehicles if self.__is_idle_state(veh.state)])

        return self.__encode_state_from_vehicles(vehicles)

    def print_state(self, t, vehicles):
        print(f"State at time {t}")
        for veh in vehicles:
            print(f"  - {veh.id}: {veh.get_cur_cz()} -> {veh.get_next_cz()} {veh.state}")

    def render(self):
        self.print_state(self.prev_timestamp, self.prev_vehicles)

    def step(self, action: int):
        veh_id = self.__decode_action_to_vehicle_id(action)
        self.sim.simulation_step_act(veh_id)

        waiting_time_sum = 0
        vehicles = []

        while True:
            timestamp, vehicles = self.sim.simulation_step_report()
            num_waiting = len(self.prev_idle_veh) - (1 if veh_id in self.prev_idle_veh else 0)
            waiting_time_sum += (timestamp - self.prev_timestamp) * num_waiting
            self.prev_timestamp = timestamp
            self.prev_vehicles = copy.deepcopy(vehicles)
            self.prev_idle_veh = set([veh.id for veh in vehicles if self.__is_idle_state(veh.state)])

            # loop until reaching a state containing at least one waiting vehicles or terminal
            if len([veh.id for veh in vehicles if veh.state == VehicleState.WAITING]) > 0 or self.sim.status != "RUNNING":
                break

        next_state = self.__encode_state_from_vehicles(vehicles)
        terminal = self.sim.status != "RUNNING"
        if terminal and self.sim.status == "DEADLOCK":
            assert(self.is_deadlock_state(next_state))
            waiting_time_sum += self.DEADLOCK_COST

        return next_state, waiting_time_sum / len(self.sim.vehicles), terminal, {}

    def __is_idle_state(self, state: VehicleState) -> bool:
        return state == VehicleState.WAITING or state == VehicleState.BLOCKED

    def __is_frontmost_vehicle(self, vehicle: Vehicle) -> bool:
        return self.sim.is_frontmost_vehicle(vehicle)

    def __encode_state_from_vehicles(self, vehicles: Iterable[Vehicle]) -> int:
        decoded_state = self.make_decoded_state()
        for veh in vehicles:
            if veh.get_cur_cz() == "^":
                if self.__is_frontmost_vehicle(veh):
                    decoded_state.src_lane_state[veh.src_lane_id].next_position = veh.get_next_cz()
                    if veh.state == VehicleState.WAITING:
                        decoded_state.src_lane_state[veh.src_lane_id].vehicle_state = "waiting"
                    elif veh.state == VehicleState.BLOCKED:
                        decoded_state.src_lane_state[veh.src_lane_id].vehicle_state = "blocked"
                    else:
                        raise Exception("GraphBasedSimEnv.__encode_state_from_vehicles: frontmost vehicle has invalid state")
                elif veh.state != VehicleState.NOT_ARRIVED:
                    decoded_state.src_lane_state[veh.src_lane_id].queue_size += 1
            elif veh.state != VehicleState.LEFT:
                veh_pos = veh.get_cur_cz()
                next_veh_pos = veh.get_next_cz()
                if veh_pos == "^":
                    if veh.state == VehicleState.WAITING:
                        decoded_state.src_lane_state[veh.src_lane_id].next_position = next_veh_pos
                elif veh_pos != "$":
                    decoded_state.cz_state[veh_pos].next_position = next_veh_pos
                    if veh.state == VehicleState.WAITING:
                        decoded_state.cz_state[veh_pos].vehicle_state = "waiting"
                    elif veh.state == VehicleState.BLOCKED:
                        decoded_state.cz_state[veh_pos].vehicle_state = "blocked"
                    elif veh.state == VehicleState.MOVING:
                        decoded_state.cz_state[veh_pos].vehicle_state = "moving"
                    else:
                        raise Exception("GraphBasedSimEnv.__encode_state_from_vehicles: vehicle has invalid state")
            else:
                assert(veh.get_cur_cz() == "$")

        return self.encode_state(decoded_state)

    def __decode_action_to_vehicle_id(self, action: int) -> str:
        decoded_action: PositionBasedStateEnv.DecodedAction = self.decode_action(action)
        vehicle = None
        
        if decoded_action.type == "src":
            vehicle = self.sim.get_waiting_veh_of_src_lane(decoded_action.id)
        elif decoded_action.type == "cz":
            vehicle = self.sim.get_waiting_veh_on_cz(decoded_action.id)
        
        return vehicle.id if vehicle is not None else ""
