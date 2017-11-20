from keras.models import Sequential
from keras.layers import Dense
import numpy as np

num_actions = 100


# given a state returns noisy matrix of actions
class Actor():
    def __init__(self, input_dim, output_dim):
        self.nn = Sequential()
        # +1 for the noisy feature
        self.nn.add(Dense(1024, input_dim=input_dim + 1))
        self.nn.add(Dense(1024))
        self.nn.add(Dense(output_dim, activation="tanh"))
        self.nn.compile(loss='mean_squared_error', optimizer='adam')

        # Noisy function
        self.noiser = np.vectorize( lambda x: (np.random.normal()) )
        self.one = np.array([[1]])


    def act(self, state):
        s = np.array([state])
        s = np.repeat(s, num_actions, axis=0)
        n_vector = self.noisy_vector(num_actions)
        s = np.concatenate((s,n_vector), axis=1)
        return self.nn.predict(s)

    def noisy_vector(self, size):
        s = np.array([[1]])
        s = np.repeat(s, size, axis=0)
        return self.noiser(s)

    def fit(self, state, action):
        s = np.append(state, self.one)
        self.nn.fit(np.array([s]), np.array([action]), epochs=3, verbose=0)
