import tensorflow as tf
import numpy as np
import reverb
import gym
from typing import Tuple

from tf_agents.typing.types import TensorSpec
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.environments import tf_py_environmentv
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.metrics import tf_metrics

# hyper-parameters
from config import config

class dqn_trainer(object):

    def __init__(
        self,
        env: gym.Env,
    ) -> None:
        # environment
        self.train_py_env: PyEnvironment = wrap_env(env)
        self.train_env: TFEnvironment = tf_py_environmentv(self.train_py_env)

        # spec
        self.time_step_spec: TensorSpec = self.train_env.time_step_spec
        self.observation_spec: TensorSpec = self.train_env.observation_spec
        self.action_spec: TensorSpec = self.train_env.action_spec

        # model architecture and agent
        self.q_net = self.configure_q_network()
        self.optimizer = self.configure_optimizer()
        self.dqn_agent = self.configure_agent()
        self.dqn_agent.initialize()

        # policies
        self.policy = self.dqn_agent.policy
        self.collect_policy = self.dqn_agent.collect_policy
        self.random_policy = self.configure_random_policy()

        # replay buffer
        self.replay_buffer, self.rb_observer = self.configure_replay_buffer()
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=config.batch_size,
            num_steps=2
        ).prefetch(3)
        self.rb_iter = iter(self.dataset)

        # loggings
        train_summary_writer = tf.summary.create_file_writer(
            config.log_dir, 
            flush_millis=10000
        )
        train_summary_writer.set_as_default()
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=config.collect_episodes_per_epoch),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=config.collect_episodes_per_epoch),
        ]

        # driver for data collection
        self.collect_driver = self.configure_collect_driver()

    def configure_q_network(self) -> QNetwork:
        return QNetwork(
            input_tensor_spec=self.observation_spec,
            action_spec=self.action_spec,
            conv_layer_params=config.conv_layer_params,
            fc_layer_params=config.fc_layer_params,
            dropout_layer_params=config.dropout_layer_params,
            activation_fn=config.activation_fn,
            kernel_initializer=config.kernel_initializer,
        )

    def configure_optimizer(self) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam(
            learning_rate=config.lr, 
            beta_1=config.betas[0], 
            beta_2=config.betas[1], 
            epsilon=config.eps
        )

    def configure_agent(self) -> DqnAgent:
        return DqnAgent(
            time_step_spec=self.time_step_spec,
            action_spec=self.action_spec,
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=config.td_errors_loss_fn
        )

    def configure_random_policy(self) -> RandomTFPolicy:
        return RandomTFPolicy(
            time_step_spec=self.time_step_spec,
            action_spec=self.action_spec
        )

    def configure_replay_buffer(self) -> Tuple[
        reverb_replay_buffer.ReverbReplayBuffer, 
        reverb_utils.ReverbAddTrajectoryObserver
    ]:
        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(self.dqn_agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

        table = reverb.Table(
            table_name,
            max_size=config.replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature
        )

        reverb_server = reverb.Server([table])

        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.dqn_agent.collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=reverb_server
        )

        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length=2
        )

        return replay_buffer, rb_observer

    def configure_collect_driver(self) -> DynamicEpisodeDriver:
        return DynamicEpisodeDriver(
            env=self.train_env,
            policy=self.collect_policy,
            observers=[self.rb_observer] + self.train_metrics,
            num_episodes=config.collect_episodes_per_epoch
        )
    
    def fit(self) -> None:
        # initialize step counter
        self.dqn_agent.train_step_counter.assign(0)

        # reset environment
        time_step = self.train_env.reset()

        # training loop
        for _ in range(config.num_iterations):

            # data collection
            self.collect_driver.run()

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(self.rb_iter)
            train_loss = self.dqn_agent.train(experience).loss

            step = self.dqn_agent.train_step_counter.numpy()

            if step % config.log_iterval == 0:
                print(f'[LOG] STEP {step} | LOSS {train_loss:.5f}')

            for train_metric in self.train_metrics:
                train_metric.tf_summaries(step_metrics=self.train_metrics[:2])
