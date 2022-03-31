import random

from tf_agents.environments import py_environment

from traffic_gen import random_traffic_generator


class AutoGenTrafficWrapperEnv(py_environment.PyEnvironment):
    def __init__(self, env):
        self.env_buffer = []
        self.origin_env = env
        self.current_env = None
        self.traffic_generator = random_traffic_generator(
            env.sim.intersection,
            num_iter = -1,
            max_vehicle_num = 20,
            mode = "stream"
        )

    @property
    def max_vehicle_num(self):
        return self.origin_env.max_vehicle_num
    
    @property
    def sim(self):
        return self.origin_env.sim
    
    @property
    def intersection(self):
        return self.origin_env.sim.intersection

    def decode_state(self, state):
        return self.origin_env.decode_state(state)

    def action_spec(self):
        return self.origin_env.action_spec()

    def observation_spec(self):
        return self.origin_env.observation_spec()

    def render(self):
        if self.current_env is not None:
            self.current_env.render()

    def _reset(self):
        self.current_env = self.get_new_env()
        return self.current_env._reset()

    def _step(self, action):
        time_step = self.current_env._step(action)
        if time_step.is_last():
            self._reset()
        return time_step

    def get_new_env(self):
        if self.env_buffer:
            return self.env_buffer.pop(0)

        new_sim = next(self.traffic_generator)
        time_step = self.origin_env._reset(new_sim=new_sim)
        while not time_step.is_last():
            valid_action_mask = self.origin_env.get_valid_action_mask(time_step.observation["observation"])
            valid_actions = [a for a, valid in enumerate(valid_action_mask) if valid]
            action = random.choice(valid_actions)
            time_step = self.origin_env._step(action)

        for _, env in self.origin_env.get_snapshots():
            self.env_buffer.append(env)

        return self.get_new_env()
