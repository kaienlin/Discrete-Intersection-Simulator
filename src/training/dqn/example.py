import gym
from trainer import DQNTrainer

def main():
    env = gym.make('CartPole-v0')
    dqn_trainer = DQNTrainer(env)
    dqn_trainer.fit()

if __name__ == '__main__':
    main()