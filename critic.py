from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Given a set of actions chooses the one with max expected reward

discount = 0.99
max_batch_size = 10000

class Critic():
    def __init__(self, input_dim):
        self.nn = Sequential()
        self.nn.add(Dense(64, input_dim=input_dim))
        self.nn.add(Dense(64))
        self.nn.add(Dense(1))
        self.inputs = []
        self.rewards = []
        self.num_batch = 0
        self.nn.compile(loss='mean_squared_error', optimizer='adam')

    def choose(self, actions, state):
        # create a matrix from states
        s = np.array([state])
        # Repeat it for each row of actions
        s = np.repeat(s, actions.shape[0], axis=0)
        # Concat actions with states
        input_values = np.concatenate((actions, s), axis=1)
        # Predic expected rewards
        QValues = self.nn.predict(input_values)
        return actions[np.argmax(QValues[0])]

    def observe(self, action, state, reward, done):
        self.addToRewards(action, state, reward)
        self.num_batch += 1
        if done or self.num_batch > max_batch_size:
            self.fitRewards()

    def addToRewards(self, action, state, reward):
        self.inputs.append(np.append(action,state))
        self.rewards.append(reward)

    def processDataset(self):
        end_reward = self.rewards[-1]
        for r in self.rewards:
            r *= (1-discount)
            r += discount * end_reward
        t = {"inputs":np.array(self.inputs), "rewards":np.array(self.rewards)}
        self.rewards = []
        self.inputs = []
        return t

    def fitRewards(self):
        self.num_batch = 0
        dataset = self.processDataset()
        self.nn.fit(dataset['inputs'], dataset['rewards'], epochs=10, verbose=1)


    # def observe(self, action, state, new_state, reward, done):
    #     input_values = np.append(action, state)
    #     # get the QValue for the next state, (not noisy)
    #     #QValue = self.nn.predict(np.array([input_values])) [0]
    #     QValueNext = self.nn.predict(np.array([np.append(action,new_state)])) [0]
    #     TargetQValue = (reward + (discount * QValueNext))
    #     if done:
    #         TargetQValue = reward
    #     self.nn.fit(np.array([input_values]), np.array([TargetQValue]), epochs=3, verbose=0)
    #     #self.last_reward = reward
