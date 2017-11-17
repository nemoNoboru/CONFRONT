# communicates the actor, the critic and the environment

from actor import Actor
from critic import Critic

class Confront():
    def __init__(self, state_dim, actions_dim):
        self.actor = Actor(state_dim, actions_dim)
        self.critic = Critic(state_dim + actions_dim)

    def act(self, state):
        # Generate noisy-filled actions
        noisy_actions = self.actor.act(state)
        # Choose the action with better expected reward
        choosed_action = self.critic.choose(noisy_actions, state)
        # train the actor to choose the better action
        self.actor.fit(state, choosed_action)
        return choosed_action

    def observe(self, action, state, reward):
        # train the critic to improve expected reward
        self.critic.observe(action, state, reward)
