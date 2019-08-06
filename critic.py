from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from experience_replay import ExperienceReplay
# Given a set of actions chooses the one with max expected reward

discount = 0.70


class Critic():
    def __init__(self, input_dim):
        self.nn = Sequential()
        self.nn.add(Dense(228, input_dim=input_dim))
        self.nn.add(Dense(356))
        self.nn.add(Dense(128))
        self.nn.add(Dense(1))
        self.inputs = []
        self.rewards = []
        self.nn.compile(loss='mean_squared_error', optimizer='adam')
        self.experience_replay = ExperienceReplay(9500)

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
        if done:
            self.fitRewards()

    def addToRewards(self, action, state, reward):
        self.inputs.append(np.append(action, state))
        # add 0.2 to compensate for fuel
        self.rewards.append(reward)

    def processDataset(self):

        if self.rewards[-1] is not 100:
            self.rewards[-1] = -100
        self.rewards[-1] = np.sum(self.rewards)
        
        for i in reversed(range(1, len(self.rewards))):
            r = self.rewards[i]
            self.rewards[i-1] = ((1 - discount) * self.rewards[i-1]) + (discount * r)
        
        for i in range(0, len(self.rewards)):
            self.experience_replay.add(self.inputs[i], self.rewards[i])

        t = {"inputs": np.array(self.inputs), "rewards": self.rewards}

        self.rewards = []
        self.inputs = []
        return t

    def fitRewards(self):
        dataset = self.processDataset()
        datasetOld = self.experience_replay.sample(5500)
        print(datasetOld)
        self.nn.fit(datasetOld['inputs'], datasetOld['rewards'], epochs=1, verbose=1)
        self.nn.fit(dataset['inputs'], dataset['rewards'], epochs=1, verbose=1)
        self.num_batch = 0


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
