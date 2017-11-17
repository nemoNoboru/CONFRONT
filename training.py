import gym
from confront import Confront
import matplotlib.pyplot as plt

env = gym.make('BipedalWalker-v2')
cft = Confront(24,4)
rewards = []

for _ in range(500):
    observation = env.reset()
    total_reward = 0
    for i in range(10):
        #env.render()
        print(env.action_space.sample())
        action = cft.act(observation)
        o, reward, done, info = env.step(action % 1)
        print("Reward is{}".format(reward))
        total_reward += reward
        cft.observe(action, observation, reward/200)
        observation = o
    rewards.append(total_reward)

plt.plot(rewards)
plt.ylabel('Rewards Confront')
plt.show()
