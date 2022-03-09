import random

import matplotlib.pyplot as plt

from traffic_gen import random_traffic_generator
from utility import read_intersection_from_json
from CP import solve_by_CP
from policy import IGreedyPolicy
from environment.vehicle_based import SimulatorEnv
from evaluate import evaluate

seed = 0
random.seed(0)

num_iter = 100
intersection = read_intersection_from_json("../intersection_configs/2x2.json")

opt_rec = []
greedy_rec = []
for n in range(4, 15):
    avg_opt = 0.0
    avg_greedy = 0.0
    env = SimulatorEnv(next(random_traffic_generator(intersection, num_iter=1, max_vehicle_num=n)), max_vehicle_num=n)
    igreedy = IGreedyPolicy(env)
    for sim in random_traffic_generator(
        intersection,
        num_iter = num_iter,
        max_vehicle_num = n,
        poisson_parameter_list = [0.5],
        mode = "eval"
    ):
        sim.run()
        avg_opt += solve_by_CP(sim)

        env.reset(new_sim=sim)
        avg_greedy += evaluate(igreedy, env)
    
    print(f"----- {n} vehicles -----")
    opt_rec.append(avg_opt / num_iter)
    greedy_rec.append(avg_greedy / num_iter)

vehicle_nums = list(range(4, 15))
plt.plot(vehicle_nums, opt_rec, label="OPT")
plt.plot(vehicle_nums, greedy_rec, label="iGreedy")
plt.ylabel("average delay per vehicle (seconds)")
plt.xlabel("maximum number of vehicles")
plt.title("traffic density = 0.5 vehicles / second")
plt.legend()
plt.savefig("CMP_OPT_GREEDY.png")
