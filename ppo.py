from tensorforce.agents import TRPOAgent
from tensorforce import Configuration
import matplotlib.pyplot as plt
import gym

env = gym.make('LunarLanderContinuous-v2')
rewards = []
print(env.action_space.sample())
# Create a Trust Region Policy Optimization agent
config = Configuration(
    batch_size=1000,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    ),
    optimization_steps=5,
    likelihood_ratio_clipping=0.2,
    discount=0.99,
    normalize_rewards=True,
    entropy_regularization=1e-2,
    #saver_spec=dict(directory='./agent', seconds=100),
)

# Create a Proximal Policy Optimization agent
agent = TRPOAgent(
    states_spec=dict(type='float', shape=(8,)),
    actions_spec=dict(type='float', shape=(2)),
    network_spec=[
        dict(type='dense', size=1024),
        dict(type='dense', size=1024),
        dict(type='dense', size=1024),
        dict(type='dense', size=1024),
        #dict(type='dense', size=128),
        #dict(type='dense', size=128),
        #dict(type='dense', size=128),
        #dict(type='dense', size=128),
        #dict(type='dense', size=128)
    ],
    config=config
)

for _ in range(50):
    observation = env.reset()
    total_reward = 0
    while True:
        action = agent.act(observation)
        o, reward, done, info = env.step(action % 1)
        print("Reward is{}".format(reward))
        total_reward += reward
        agent.observe(reward=reward/200, terminal=done)
        observation = o
        if done:
            break
    rewards.append(total_reward)

plt.plot(rewards)
plt.ylabel('Rewards PPO')
plt.show()
