import tensorflow as tf
import numpy as np
import reverb
import gym

# environment
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.environments import tf_py_environmentv

# hyper-parameters
from hparams import hparams

class dqn_trainer(object):

    def __init__(
        self,
        env: gym.Env,
    ) -> None:
        self.train_py_env = wrap_env(env)
        self.eval_py_env = wrap_env(env)
        self.train_env = tf_py_environmentv(self.train_py_env)
        self.eval_enc = tf_py_environmentv(self.eval_py_env)