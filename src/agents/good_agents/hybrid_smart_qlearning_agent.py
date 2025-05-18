import numpy as np
from src.agents.base_agent import BaseAgent

ACTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
    (0, 0)
]


class ExpectedSARSALambdaSmartAgent(BaseAgent):
    def __init__(self, learning_rate=0.1, discount_factor=0.99, exploration_rate=0.3, lambda_=0.8):
        super().__init__()
        self.q_table = {}
        self.eligibility = {}
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.lambda_ = lambda_
        self.np_random = np.random.default_rng()
        self.goal = None

        self.position_bins = 8
        self.wind_bins = 8

    def set_goal(self, goal):
        self.goal = np.array(goal)

    def discretize_state(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]
        windfield = np.array(observation[6:]).reshape(-1, 2)

        x_bin = min(int(x / 32 * self.position_bins), self.position_bins - 1)
        y_bin = min(int(y / 32 * self.position_bins), self.position_bins - 1)

        avg_wind = np.mean(windfield, axis=0)
        avg_wind_dir = np.arctan2(avg_wind[1], avg_wind[0])
        avg_wind_bin = int(((avg_wind_dir + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins

        local_wind_dir = np.arctan2(wy, wx)
        local_wind_bin = int(
            ((local_wind_dir + np.pi) / (2 * np.pi) * self.wind_bins)
        ) % self.wind_bins

        v_mag = np.linalg.norm([vx, vy])
        if v_mag < 0.1:
            v_dir_bin = 0
        else:
            v_dir = np.arctan2(vy, vx)
            v_dir_bin = int(((v_dir + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins

        if self.goal is not None:
            goal_vec = self.goal - np.array([x, y])
            goal_dist = np.linalg.norm(goal_vec)
            goal_dir = np.arctan2(goal_vec[1], goal_vec[0])
            goal_dir_bin = int(((goal_dir + np.pi) / (2 * np.pi) * self.wind_bins)) % self.wind_bins
            dist_bin = min(int(goal_dist / 10), 3)
        else:
            goal_dir_bin = 0
            dist_bin = 0

        return (
            x_bin, y_bin,
            avg_wind_bin, local_wind_bin,
            v_dir_bin,
            goal_dir_bin, dist_bin
        )

    def smart_action(self, observation):
        wx, wy = observation[4], observation[5]
        wind = np.array([wx, wy])
        wind_norm = np.linalg.norm(wind)
        if wind_norm < 1e-3:
            return 0
        wind_dir = wind / wind_norm

        if self.goal is None:
            return 0
        pos = np.array([observation[0], observation[1]])
        goal_vec = self.goal - pos
        goal_dir = goal_vec / (np.linalg.norm(goal_vec) + 1e-8)

        best_score = -np.inf
        best_action = 0
        for idx, (dx, dy) in enumerate(ACTIONS[:-1]):
            move = np.array([dx, dy])
            move_dir = move / (np.linalg.norm(move) + 1e-8)
            angle_to_wind = np.arccos(np.clip(np.dot(move_dir, wind_dir), -1, 1))
            angle_deg = np.degrees(angle_to_wind)
            if angle_deg < 45:
                continue
            eff = np.cos(np.radians(abs(angle_deg - 90)))
            to_goal = np.dot(move_dir, goal_dir)
            score = 0.6 * to_goal + 0.4 * eff
            if score > best_score:
                best_score = score
                best_action = idx
        return best_action

    def act(self, observation):
        state = self.discretize_state(observation)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if self.np_random.random() < self.epsilon:
            return self.np_random.integers(0, 9)
        else:
            return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)
        if state not in self.eligibility:
            self.eligibility[state] = np.zeros(9)

        q_next = self.q_table[next_state]
        best_action = np.argmax(q_next)
        policy = np.full(9, self.epsilon / 9)
        policy[best_action] += 1.0 - self.epsilon
        expected_q = np.dot(policy, q_next)

        td_target = reward + self.gamma * expected_q
        td_error = td_target - self.q_table[state][action]

        self.eligibility[state][action] += 1

        for s in self.q_table:
            self.q_table[s] += self.alpha * td_error * self.eligibility.get(s, np.zeros(9))
            self.eligibility[s] = self.gamma * self.lambda_ * self.eligibility.get(s, np.zeros(9))

    def reset(self):
        self.goal = None
        self.eligibility.clear()

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
