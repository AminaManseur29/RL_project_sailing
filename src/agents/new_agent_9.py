import numpy as np
from agents.base_agent import BaseAgent


class QLearningWindAgent(BaseAgent):
    """
    Q-learning agent that learns best actions from interaction.
    """

    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.2):
        super().__init__()
        self.goal = np.array([16, 31])
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # key: (x, y), value: [q-values per action]

    def get_state(self, observation):
        x, y = int(observation[0]), int(observation[1])
        return (x, y)

    def act(self, observation):
        state = self.get_state(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        if np.random.rand() < self.epsilon:
            return np.random.choice(9)
        else:
            return int(np.argmax(self.q_table[state]))

    def update(self, observation, action, reward, next_observation):
        state = self.get_state(observation)
        next_state = self.get_state(next_observation)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.lr * td_error

    def reset(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path, allow_pickle=True).item()
