# holds a batch of experiences to be replayed
import numpy as np


class ExperienceReplay():
    def __init__(self, s):
        self.inputs = []
        self.rewards = []
        self.max_size = s
        self.size = 0

    # TODO Optimize this (shit) to allow for batch add
    def add(self, inpu, reward):
        if self.max_size < self.size:
            self.removeOne()

        self.inputs.append(inpu)
        self.rewards.append(reward)
        self.size += 1

    def sample(self, size):
        i = []
        r = []
        if size > self.size :
            size = self.size
        for _ in range(size):
            t = np.random.randint(self.size)
            i.append(self.inputs[t])
            r.append(self.rewards[t])
        return {"inputs": np.array(i), "rewards": np.array(r)}

    def removeOne(self):
        #minimal = np.argmin(np.array(self.rewards))
        minimal = np.random.randint(self.size)
        del self.inputs[minimal]
        del self.rewards[minimal]
        self.size -= 1
