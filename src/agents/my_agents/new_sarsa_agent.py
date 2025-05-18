import numpy as np                # type: ignore
from agents.base_agent import BaseAgent


class SARSAAgent1(BaseAgent):
    """ Enhance the Q-Learning Agent + SARSA with intelligent exploration"""
    def __init__(self, discount_factor=0.9):
        super().__init__()
        self.np_random = np.random.default_rng()

        # Learning parameters
        self.discount_factor = discount_factor

        # State discretization parameters
        self.position_bins = 8     # Discretize the grid into 8x8
        self.velocity_bins = 4     # Discretize velocity into 4 bins
        self.wind_bins = 8         # Discretize wind directions into 8 bins

        # Initialize Q-table
        # State space: position_x, position_y, velocity_direction, wind_direction
        # Action space: 9 possible actions
        self.q_table = {}
        self.N = {}

    def discretize_state(self, observation):
        """Convert continuous observation to discrete state for Q-table lookup."""
        # Extract position, velocity and wind from observation
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        windfield = observation[6:]  # flattened wind field

        # Discretize position (assume 32x32 grid)
        grid_size = 32
        x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)

        # Discretize velocity direction (ignoring magnitude for simplicity)
        v_magnitude = np.sqrt(vx**2 + vy**2)
        if v_magnitude < 0.1:  # If velocity is very small, consider it as a separate bin
            v_bin = 0
        else:
            v_direction = np.arctan2(vy, vx)  # Range: [-pi, pi]
            v_bin = int(
                ((v_direction + np.pi) / (2 * np.pi) * (self.velocity_bins-1)) + 1
            ) % self.velocity_bins

        # Discretize wind direction
        wind_direction = np.arctan2(wy, wx)  # Range: [-pi, pi]
        wind_bin = int(((wind_direction + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins

        # Extraire direction moyenne du champ de vent global
        windfield = np.array(windfield).reshape(-1, 2)
        avg_wx, avg_wy = np.mean(windfield[:, 0]), np.mean(windfield[:, 1])
        avg_wind_dir = int(
            ((np.arctan2(avg_wy, avg_wx) + np.pi) / (2 * np.pi) * self.wind_bins)
        ) % self.wind_bins

        # Return discrete state tuple
        return (x_bin, y_bin, v_bin, wind_bin, avg_wind_dir)

    def should_explore(self, state):
        # Estime epsilon comme 1 / min(N) pour les actions déjà explorées
        counts = [self.N.get((state, a), 0) for a in range(9)]
        nonzero_counts = [c for c in counts if c > 0]

        if not nonzero_counts:
            return True  # explore à 100% si jamais visité

        min_count = min(nonzero_counts)
        epsilon_adaptive = 1 / min_count
        return self.np_random.random() < epsilon_adaptive

    def act(self, observation):
        """Choose an action using epsilon-greedy policy."""
        # Discretize the state
        state = self.discretize_state(observation)

        # Epsilon-greedy action selection
        if self.should_explore(state):
            # Explore: choose a random action
            return self.np_random.integers(0, 9)
        else:
            # Exploit: choose the best action according to Q-table
            if state not in self.q_table:
                # If state not in Q-table, initialize it
                self.q_table[state] = np.zeros(9)

            # Return action with highest Q-value
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, next_action):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        if (state, action) not in self.N:
            self.N[(state, action)] = 0
        self.N[(state, action)] += 1

        # TD target with SARSA
        td_target = reward + self.discount_factor * self.q_table[next_state][next_action]
        td_error = td_target - self.q_table[state][action]

        alpha = 1 / self.N[(state, action)]  # Décroissance dynamique
        self.q_table[state][action] += alpha * td_error

    def reset(self):
        """Reset the agent for a new episode."""
        # Nothing to reset for Q-learning agent
        pass

    def seed(self, seed=None):
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)

    def save(self, path):
        """Save the Q-table to a file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        """Load the Q-table from a file."""
        import pickle
        with open(path, 'rb') as f:
            self.q_table = pickle.load(f)

