from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Given a set of actions chooses the one with max expected reward

learning_rate = 0.1

class Critic():
    def __init__(self, input_dim):
        self.nn = Sequential()
        self.nn.add(Dense(64, input_dim=input_dim))
        self.nn.add(Dense(64))
        self.nn.add(Dense(1))
        self.last_reward = 0
        self.nn.compile(loss='mean_squared_error', optimizer='adam')

    def choose(self, actions, state):
        input_values = []
        for action in actions:
            input_values.append( np.append(action , state) )
        QValues = self.nn.predict(np.array(input_values))[0]
        return actions[np.argmax(QValues)]

    def observe(self, action, state, reward):
        input_values = np.append(action, state)
        #reward = self.last_reward + (learning_rate * reward)
        self.nn.fit(np.array([input_values]), np.array([reward]), epochs=3, verbose=1)
        #self.last_reward = reward
