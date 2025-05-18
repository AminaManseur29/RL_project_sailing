"""
SARSA Lambda Agent for the Sailing Challenge, using TD(lambda) instead of TD(0)

This file provides a simple agent that always goes north.
Students can use it as a reference for how to implement a sailing agent.
"""

import numpy as np                # type: ignore
from agents.base_agent import BaseAgent


class ExpectedSARSALambdaAgent(BaseAgent):
    def __init__(self, learning_rate=0.1, gamma=0.9, epsilon=0.1, lambda_=0.8):
        super().__init__()
        self.alpha = learning_rate
        self.lambda_ = lambda_
        self.gamma = gamma
        self.epsilon = epsilon

        self.np_random = np.random.default_rng()

        self.position_bins = 8
        self.velocity_bins = 4
        self.wind_bins = 8

        self.q_table = {}
        self.eligibility = {}

    def discretize_state(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        windfield = np.array(observation[6:]).reshape(-1, 2)

        grid_size = 32
        x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)

        v_mag = np.linalg.norm([vx, vy])
        v_bin = 0 if v_mag < 0.1 else int(
            ((np.arctan2(vy, vx) + np.pi) / (2 * np.pi) * (self.velocity_bins - 1)) + 1
        ) % self.velocity_bins

        wind_dir = np.arctan2(wy, wx)
        wind_bin = int(
            ((wind_dir + np.pi) / (2 * np.pi) * self.wind_bins)
        ) % self.wind_bins

        avg_wind = np.mean(windfield, axis=0)
        avg_wind_dir = int(
            ((np.arctan2(avg_wind[1], avg_wind[0]) + np.pi) / (2 * np.pi) * self.wind_bins)
        ) % self.wind_bins

        return (x_bin, y_bin, v_bin, wind_bin, avg_wind_dir)

    def act(self, observation):
        state = self.discretize_state(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        if self.np_random.random() < self.epsilon:
            return self.np_random.integers(0, 9)
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        # Expected SARSA update (policy is epsilon-greedy)
        q_next = self.q_table[next_state]
        best_action = np.argmax(q_next)
        policy = np.full(9, self.epsilon / 9)
        policy[best_action] += 1.0 - self.epsilon
        expected_q = np.dot(policy, q_next)

        # TD error
        delta = reward + self.gamma * expected_q - self.q_table[state][action]

        # Eligibility update
        for s in self.q_table:
            for a in range(9):
                self.eligibility.setdefault(s, np.zeros(9))
                self.eligibility[s][a] *= self.gamma * self.lambda_

        self.eligibility.setdefault(state, np.zeros(9))
        self.eligibility[state][action] += 1

        # Q-table update
        for s in self.q_table:
            for a in range(9):
                self.q_table[s][a] += self.alpha * delta * self.eligibility[s][a]

    def reset(self):
        self.eligibility = {}  # reset eligibility traces between episodes

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)
