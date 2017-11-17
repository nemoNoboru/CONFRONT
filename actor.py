from keras.models import Sequential
from keras.layers import Dense
import numpy as np

num_actions = range(40)


# given a state returns noisy matrix of actions
class Actor():
    def __init__(self, input_dim, output_dim):
        self.nn = Sequential()
        # +1 for the noisy feature
        self.nn.add(Dense(1024, input_dim=input_dim))
        self.nn.add(Dense(1024))
        self.nn.add(Dense(output_dim, activation="tanh"))
        self.nn.compile(loss='mean_squared_error', optimizer='adam')


    def act(self, state):
        l = []
        # Warning, highly inefficient code here
        #l.append(state)
        #for _ in num_actions:
        #    noisy_state = [x + (np.random.normal() * 0.01) for x in state ]
        #    l.append(noisy_state)
        #noisy_states = np.array(l)
        action = self.nn.predict(np.array([state]))
        for _ in num_actions:
            noisy_actions = [x + (np.random.normal() * 0.01) for x in action ]
            l.append(noisy_state)
        return l

    def fit(self, state, action):
        self.nn.fit(np.array([state]), np.array([action]), epochs=3, verbose=1)
