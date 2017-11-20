import gym
from confront import Confront
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('MountainCarContinuous-v0')
cft = Confront(2,1)
rewards = []

# set seeds to 0
env.seed(0)
np.random.seed(0)

for j in range(1500):
    observation = env.reset()
    total_reward = 0
    for i in range(1000):
        #env.render()
        #print(env.action_space.sample())
        action = cft.act(observation)
        o, reward, done, info = env.step(action % 1)
        #print("Reward is{}".format(reward))
        total_reward += reward
        cft.observe(action, observation, reward, done)
        observation = o
        if done:
            break
    rewards.append(total_reward)
    print("Episode num: {} total reward: {} num steps per episode: {}".format(j, total_reward, i))

plt.plot(rewards)
plt.ylabel('Rewards Confront')
plt.show()
