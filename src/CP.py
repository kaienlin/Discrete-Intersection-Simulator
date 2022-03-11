from ortools.sat.python import cp_model
from collections import defaultdict
import copy

from simulation import Simulator
from simulation.tcg import TimingConflictGraph, EdgeType
from utility import read_intersection_from_json

def calculate_passing_time_sum_upper_bound(sim: Simulator) -> int:
    last_leaving_time = 0
    max_edge_waiting_time = max([edge.waiting_time for edge in sim.TCG.E]) if len(sim.TCG.E) > 0 else 1
    for vehicle in sorted(sim.vehicles, key=lambda vehicle: vehicle.earliest_arrival_time):
        entering_time = max(vehicle.earliest_arrival_time, last_leaving_time)
        leaving_time = entering_time + vehicle.vertex_passing_time * len(vehicle.trajectory)
        leaving_time += max_edge_waiting_time * len(vehicle.trajectory)
        last_leaving_time = leaving_time
    return last_leaving_time

def solve_by_CP(sim: Simulator):
    if len(sim.vehicles) == 0:
        return 0

    model = cp_model.CpModel()
    tcg: TimingConflictGraph = copy.deepcopy(sim.TCG)
    tcg.add_unsure_type3_edges()
    passing_time_sum_upper_bound = calculate_passing_time_sum_upper_bound(sim)

    vertex_to_vars = {}
    for vertex in tcg.V:
        suffix = f"{vertex.vehicle.id}_{vertex.cz_id}"
        entering_time_var = model.NewIntVar(0, passing_time_sum_upper_bound, f"entering_time_{suffix}")
        leaving_time_var = model.NewIntVar(0, passing_time_sum_upper_bound, f"leaving_time_{suffix}")
        vertex_to_vars[vertex.vehicle.id, vertex.cz_id] = (entering_time_var, leaving_time_var)
    
    edge_to_vars = {}
    for edge in tcg.E:
        if edge.type != EdgeType.TYPE_3:
            continue
        selected_var = model.NewBoolVar(f"edge_selected_{edge.id}")
        edge_to_vars[edge.id] = selected_var

    processed_edge_ids = set()
    for edge in tcg.E:
        if edge.type != EdgeType.TYPE_3 or edge.id in processed_edge_ids:
            continue
        selected_var = edge_to_vars[edge.id]
        alternative_edge = [e for e in tcg.E if e.v_from.id == edge.v_to.id and e.v_to.id == edge.v_from.id][0]
        disjunctive_var = edge_to_vars[alternative_edge.id]
        processed_edge_ids.add(alternative_edge.id)
        model.AddBoolXOr([selected_var, disjunctive_var])

    for vertex in tcg.V:
        entering_time_var, leaving_time_var = vertex_to_vars[vertex.vehicle.id, vertex.cz_id]
        model.Add(leaving_time_var >= entering_time_var + vertex.passing_time)
        for in_e in vertex.in_edges:
            parent = in_e.v_from
            p_entering_time_var, p_leaving_time_var = vertex_to_vars[parent.vehicle.id, parent.cz_id]
            if in_e.type == EdgeType.TYPE_1:
                model.Add(p_leaving_time_var == entering_time_var)
            elif in_e.type == EdgeType.TYPE_2:
                model.Add(entering_time_var >= p_leaving_time_var + in_e.waiting_time)
            elif in_e.type == EdgeType.TYPE_3:
                selected_var = edge_to_vars[in_e.id]
                model.Add(entering_time_var >= p_leaving_time_var + in_e.waiting_time).OnlyEnforceIf(selected_var)

    
    for vehicle in sim.vehicles:
        entering_time_var, leaving_time_var = vertex_to_vars[vehicle.id, vehicle.trajectory[0]]
        model.Add(entering_time_var >= vehicle.earliest_arrival_time)
    
    vehicle_passing_time_list = []
    for vehicle in sim.vehicles:
        entering_time_var, leaving_time_var = vertex_to_vars[vehicle.id, vehicle.trajectory[-1]]
        vehicle_passing_time_list.append(leaving_time_var - vehicle.earliest_arrival_time)

    model.Minimize(sum(vehicle_passing_time_list))
            
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    tot = 0
    if status == cp_model.OPTIMAL:
        for vehicle in sim.vehicles:
            #print(vehicle.id, ": ", end="")
            leaving_last_cz_time = 0
            for cz_id in vehicle.trajectory:
                entering_time_var, leaving_time_var = vertex_to_vars[vehicle.id, cz_id]
                leaving_last_cz_time = solver.Value(leaving_time_var)
                #print(solver.Value(entering_time_var), leaving_last_cz_time, end=",")
            tot += leaving_last_cz_time - vehicle.earliest_arrival_time - vehicle.vertex_passing_time * len(vehicle.trajectory)
            #print("")

    return tot / len(sim.vehicles) / 10

if __name__ == "__main__":
    intersection = read_intersection_from_json("../intersection_configs/2x2.json")
    sim = Simulator(intersection)
    sim.load_traffic("./tmp/0.json")
    sim.run()
    print(f"{solve_by_CP(sim):.4f}")
