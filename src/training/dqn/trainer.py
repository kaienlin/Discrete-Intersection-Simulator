import tensorflow as tf
import numpy as np
import os
import reverb
from datetime import datetime

from tf_agents.typing.types import TensorSpec
from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments.tf_environment import TFEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.networks.q_network import QNetwork
from tf_agents.agents import TFAgent
from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.policies.py_tf_eager_policy import PyTFEagerPolicy
from tf_agents.policies.q_policy import QPolicy
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.drivers import py_driver
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common
from tf_agents.specs import tensor_spec

# hyper-parameters
from training.dqn.config import config

from environment.func_approx import AutoGenTrafficWrapperEnv
from evaluate import batch_evaluate_tf
from traffic_gen import datadir_traffic_generator


def observation_and_action_constraint_splitter(observation):
    return observation["observation"], observation["valid_actions"]

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

class DQNTrainer(object):

    def __init__(
        self,
        env: PyEnvironment,
    ) -> None:
        # model name
        now = datetime.now()
        time = now.strftime("%m-%d")
        self.model_root = f'output-{time}'
        self.ckpt_dir = os.path.join(self.model_root, 'checkpoints')
        self.log_dir = os.path.join(self.model_root, 'loggings')

        if isinstance(env, AutoGenTrafficWrapperEnv):
            self.intersection = env.intersection
            self.max_vehicle_num = env.max_vehicle_num
            self.observation_decoder = env.decode_state

        # environment
        self.train_py_env: PyEnvironment = env
        self.train_env: TFEnvironment = TFPyEnvironment(env)
        self.eval_env: TFEnvironment = TFPyEnvironment(env)

        # spec
        self.time_step_spec: TensorSpec = self.train_env.time_step_spec()
        self.observation_spec: TensorSpec = self.train_env.observation_spec()
        self.action_spec: TensorSpec = self.train_env.action_spec()

        # model architecture and agent
        self.step = tf.Variable(0, dtype=tf.int64)
        self.q_net = self.configure_q_network()
        self.optimizer = self.configure_optimizer()
        self.agent = self.configure_agent()
        self.agent.initialize()

        # policies
        self.policy = self.agent.policy
        self.collect_policy = self.agent.collect_policy
        self.random_policy = self.configure_random_policy()

        # replay buffer
        self.replay_buffer, self.rb_observer = self.configure_replay_buffer()
        self.dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=config.batch_size,
            num_steps=2
        )
        self.rb_iter = iter(self.dataset)

        # loggings
        train_summary_writer = tf.summary.create_file_writer(
            self.log_dir, 
            flush_millis=10000
        )
        train_summary_writer.set_as_default()
        self.train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(buffer_size=config.collect_steps_per_iteration),
            tf_metrics.AverageEpisodeLengthMetric(buffer_size=config.collect_steps_per_iteration),
        ]

        # driver for data collection
        self.initial_driver = self.configure_initial_driver()
        self.collect_driver = self.configure_collect_driver()
        self.initial_driver.run(self.train_py_env.reset())

        # checkpointer
        self.train_checkpointer = self.configure_checkpointer()
        self.train_checkpointer.initialize_or_restore()

        # checkpointer
        self.train_checkpointer = self.configure_checkpointer()
        self.train_checkpointer.initialize_or_restore()

    def configure_q_network(self) -> QNetwork:
        return QNetwork(
            input_tensor_spec=self.observation_spec["observation"],
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

    def configure_agent(self) -> TFAgent:
        if not config.use_ddqn:
            return DqnAgent(
                time_step_spec=self.time_step_spec,
                action_spec=self.action_spec,
                q_network=self.q_net,
                optimizer=self.optimizer,
                epsilon_greedy=config.epsilon_greedy,
                gamma=config.gamma,
                target_update_tau=config.target_update_tau,
                target_update_period=config.target_update_period,
                td_errors_loss_fn=config.td_errors_loss_fn,
                train_step_counter=self.step,
                observation_and_action_constraint_splitter=observation_and_action_constraint_splitter
            )
        else:
            return DdqnAgent(
                time_step_spec=self.time_step_spec,
                action_spec=self.action_spec,
                q_network=self.q_net,
                optimizer=self.optimizer,
                epsilon_greedy=config.epsilon_greedy,
                gamma=config.gamma,
                target_update_tau=config.target_update_tau,
                target_update_period=config.target_update_period,
                td_errors_loss_fn=config.td_errors_loss_fn,
                train_step_counter=self.step,
                observation_and_action_constraint_splitter=observation_and_action_constraint_splitter
            )

    def configure_random_policy(self) -> RandomTFPolicy:
        return RandomTFPolicy(
            time_step_spec=self.time_step_spec,
            action_spec=self.action_spec,
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter,
        )

    def configure_eval_policy(self) -> QPolicy:
        return QPolicy(
            time_step_spec=self.time_step_spec,
            action_spec=self.action_spec,
            q_network=self.q_net,
            observation_and_action_constraint_splitter=observation_and_action_constraint_splitter
        )

    def configure_replay_buffer(self):
        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(self.agent.collect_data_spec)
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
            self.agent.collect_data_spec,
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

    def configure_initial_driver(self) -> py_driver.PyDriver:
        return py_driver.PyDriver(
            env=self.train_py_env,
            policy=PyTFEagerPolicy(
                self.random_policy, 
                use_tf_function=True
            ),
            observers=[self.rb_observer],
            max_steps=config.initial_collect_steps
        )

    def configure_collect_driver(self) -> py_driver.PyDriver:
        return py_driver.PyDriver(
            env=self.train_py_env,
            policy=PyTFEagerPolicy(
                self.collect_policy,
                use_tf_function=True
            ),
            observers=[self.rb_observer],
            max_steps=config.collect_steps_per_iteration
        )
    
    def configure_checkpointer(self) -> common.Checkpointer:
        return common.Checkpointer(
            ckpt_dir=self.ckpt_dir,
            max_to_keep=config.ckpt_kept_num,
            agent=self.agent,
            policy=self.policy,
            replay_buffer=self.replay_buffer,
            global_step=self.step
        )

    def fit(self) -> None:
        # reset environment
        time_step = self.train_py_env.reset()
        tot_loss = 0

        # training loop
        for _ in range(config.num_iterations):

            # data collection
            time_step, _ = self.collect_driver.run(time_step)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(self.rb_iter)
            train_loss = self.agent.train(experience).loss
            tot_loss += train_loss

            step = self.agent.train_step_counter.numpy()

            if step % config.log_iterval == 0:
                print(f'[LOG] STEP {step} | LOSS {tot_loss / config.log_iterval:.5f}')
                tot_loss = 0

            if step % config.valid_interval == 0:
                # avg_return = compute_avg_return(self.eval_env, self.agent.policy)
                # print(f'[VALID] Average Reward: {avg_return:.5f}')
                sim_gen = datadir_traffic_generator(self.intersection, config.valid_data_dir)
                avg_reward = batch_evaluate_tf(self.configure_eval_policy(), sim_gen, self.max_vehicle_num)
                print(f"Validation Reward = {avg_reward}")

            if step % config.ckpt_interval == 0:
                self.train_checkpointer.save(step)

            # for train_metric in self.train_metrics:
            #     train_metric.tf_summaries(step_metrics=self.train_metrics[:2])
