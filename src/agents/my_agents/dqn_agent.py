"""
Deep Q-Learning Agent for the Sailing Challenge
"""


import numpy as np                # type: ignore
import random
import torch
import torch.nn as nn
import torch.optim as optim
from agents.base_agent import BaseAgent


class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim,
        action_dim=9,
        lr=1e-3, gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = []

    def preprocess_state(self, obs):
        return torch.FloatTensor(obs).to(self.device)

    def act(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = self.preprocess_state(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return int(torch.argmax(q_values).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = zip(*random.sample(self.memory, batch_size))
        states, actions, rewards, next_states, dones = map(np.array, batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        q_values = self.q_net(states)
        next_q_values = self.q_net(next_states)

        target_q = q_values.clone()
        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * torch.max(next_q_values[i])
            target_q[i][actions[i]] = target

        loss = self.criterion(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def reset(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
