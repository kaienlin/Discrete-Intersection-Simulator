from utility import read_intersection_from_json
from simulation import Simulator
from environment.func_approx import MinimumEnv, AutoGenTrafficWrapperEnv
from training.dqn import DQNTrainer

intersection = read_intersection_from_json("../intersection_configs/2x2.json")
sim = Simulator(intersection)
env = AutoGenTrafficWrapperEnv(MinimumEnv(sim, 8))
trainer = DQNTrainer(env)
trainer.fit()
