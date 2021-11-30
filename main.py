import simulator
import random

def get_4cz_intersection():
    '''
                   N
            ---------------
            |  1   |  2   |
            |      |      |
        W   ---------------   E
            |  3   |  4   |
            |      |      |
            ---------------
                   S
    '''
    I = simulator.Intersection()
    for cz_id in range(1, 5):
        I.add_conflict_zone(str(cz_id))
    
    I.add_src_lane("N", ["1"])
    I.add_src_lane("E", ["2"])
    I.add_src_lane("W", ["3"])
    I.add_src_lane("S", ["4"])

    I.add_dst_lane("N", ["2"])
    I.add_dst_lane("E", ["4"])
    I.add_dst_lane("W", ["1"])
    I.add_dst_lane("S", ["3"])

    I.add_trajectory(("1"))
    I.add_trajectory(("1", "3"))
    I.add_trajectory(("1", "3", "4"))

    I.add_trajectory(("2"))
    I.add_trajectory(("2", "1"))
    I.add_trajectory(("2", "1", "3"))

    I.add_trajectory(("3"))
    I.add_trajectory(("3", "4"))
    I.add_trajectory(("3", "4", "2"))

    I.add_trajectory(("4"))
    I.add_trajectory(("4", "2"))
    I.add_trajectory(("4", "2", "1"))

    return I

def print_state(t, vehicles):
    print(f"State at time {t}")
    for veh in vehicles:
        print(f"  - {veh.id}: {veh.get_cur_cz()} {veh.state}")

if __name__ == "__main__":
    random.seed(0)
    intersection = get_4cz_intersection()
    sim = simulator.Simulator(intersection)
    sim.add_vehicle("a", 0, ("1", "3", "4"), "N", "E")
    sim.add_vehicle("b", 0, ("4", "2", "1"), "S", "W")
    sim.run()

    for _ in range(4):
        timestamp, vehicles = sim.simulation_step_report()
        print_state(timestamp, vehicles)
        sim.simulation_step_act(["a", "b"][_ % 2])

    sim.add_vehicle("c", 5, ("2", "1", "3"), "E", "S")
    #sim.print_TCG()
    while True:
        timestamp, vehicles = sim.simulation_step_report()
        if sim.status != "RUNNING":
            break
        print_state(timestamp, vehicles)
        sim.simulation_step_act(random.choice(["a", "b", "c"]))
    
    sim.print_TCG()
    print(sim.status)
