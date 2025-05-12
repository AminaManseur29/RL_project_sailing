"""
Special agent for the Sailing Challenge
"""


import numpy as np                # type: ignore
from agents.base_agent import BaseAgent

ACTIONS = [
    (0, 1),    # 0: N
    (1, 1),    # 1: NE
    (1, 0),    # 2: E
    (1, -1),   # 3: SE
    (0, -1),   # 4: S
    (-1, -1),  # 5: SW
    (-1, 0),   # 6: W
    (-1, 1),   # 7: NW
    (0, 0),    # 8: Stay
]


class SailingPhysicsAgent(BaseAgent):
    """Physics-Based Approaches"""
    def __init__(self):
        self.goal = None  # to be set on first call
        self.tack_side = 1  # +1 for right tack, -1 for left tack

    def reset(self):
        self.tack_side *= -1  # change tack direction to alternate strategy
        self.goal = None

    def seed(self, seed):
        np.random.seed(seed)

    def set_goal(self, goal):
        self.goal = np.array(goal)

    def act(self, observation):
        if self.goal is None:
            # fallback: center of the map
            self.goal = np.array([16, 16])

        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        pos = np.array([x, y])
        wind = np.array([wx, wy])

        goal_vector = self.goal - pos
        if np.linalg.norm(goal_vector) == 0:
            return 8  # stay

        goal_dir = goal_vector / np.linalg.norm(goal_vector)
        wind_dir = wind / (np.linalg.norm(wind) + 1e-8)

        angle_to_wind = np.arccos(np.clip(np.dot(goal_dir, wind_dir), -1, 1))
        angle_deg = np.degrees(angle_to_wind)

        if angle_deg < 45:
            # Target is against the wind — tack
            tack_angle = np.radians(45) * self.tack_side
            rotation_matrix = np.array([
                [np.cos(tack_angle), -np.sin(tack_angle)],
                [np.sin(tack_angle),  np.cos(tack_angle)]
            ])
            target_direction = rotation_matrix @ wind_dir
        else:
            # Go toward the goal directly
            target_direction = goal_dir

        # Choisir l'action la plus proche du vecteur cible
        best_action = 8
        best_score = -np.inf
        for idx, a in enumerate(ACTIONS[:-1]):  # exclude "stay" from primary selection
            direction = np.array(a) / (np.linalg.norm(a) + 1e-8)
            score = np.dot(direction, target_direction)
            if score > best_score:
                best_score = score
                best_action = idx

        return best_action


class HybridPhysicsQLearningAgent(BaseAgent):
    def __init__(self, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.3):
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.np_random = np.random.default_rng()

        self.position_bins = 8
        self.velocity_bins = 4
        self.wind_bins = 8

    def discretize_state(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        windfield = observation[6:]

        grid_size = 32
        x_bin = min(int(x / grid_size * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / grid_size * self.position_bins), self.position_bins - 1)

        v_mag = np.sqrt(vx**2 + vy**2)
        if v_mag < 0.1:
            v_bin = 0
        else:
            v_dir = np.arctan2(vy, vx)
            v_bin = int(
                ((v_dir + np.pi) / (2 * np.pi) * (self.velocity_bins - 1)) + 1
            ) % self.velocity_bins

        wind_dir = np.arctan2(wy, wx)
        wind_bin = int(((wind_dir + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins

        windfield = np.array(windfield).reshape(-1, 2)
        avg_wind_dir = np.arctan2(np.mean(windfield[:, 1]), np.mean(windfield[:, 0]))
        avg_wind_bin = int(((avg_wind_dir + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins

        return (x_bin, y_bin, v_bin, wind_bin, avg_wind_bin)

    def act(self, observation):
        state = self.discretize_state(observation)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        if self.np_random.random() < self.epsilon:
            # Exploration sur actions valides seulement
            valid_actions = self.get_valid_actions(observation)
            return int(self.np_random.choice(valid_actions))

        # Exploitation : parmi les meilleures actions, ne garder que celles valides
        q_vals = self.q_table[state]
        sorted_actions = np.argsort(q_vals)[::-1]  # tri décroissant
        for a in sorted_actions:
            if a in self.get_valid_actions(observation):
                return int(a)

        return 8  # fallback : rester sur place

    def get_valid_actions(self, observation):
        wx, wy = observation[4], observation[5]
        wind_dir = np.arctan2(wy, wx)

        valid = []
        for idx, (dx, dy) in enumerate(ACTIONS[:-1]):  # exclude "stay"
            move_dir = np.arctan2(dy, dx)
            angle = np.abs(move_dir - wind_dir)
            angle = angle if angle <= np.pi else (2*np.pi - angle)
            angle_deg = np.degrees(angle)

            if angle_deg >= 45:  # pas trop face au vent
                valid.append(idx)

        valid.append(8)  # rester est toujours valide
        return valid

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def reset(self):
        pass

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
