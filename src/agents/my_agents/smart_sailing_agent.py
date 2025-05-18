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


class SmartSailingAgent(BaseAgent):
    """
    An improved deterministic sailing agent using sailing physics heuristics.
    """

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        self.goal = None

    def set_goal(self, goal):
        self.goal = np.array(goal)

    def act(self, observation: np.ndarray) -> int:
        if self.goal is None:
            return 0  # Fallback to North

        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        wind_vec = np.array([wx, wy])
        wind_norm = np.linalg.norm(wind_vec)

        # Skip if no wind
        if wind_norm < 1e-3:
            return 0  # Go North by default

        wind_dir = wind_vec / wind_norm

        # 🎯 Calculer la direction vers l'objectif
        current_pos = np.array([x, y])
        goal_vec = self.goal - current_pos
        goal_dist = np.linalg.norm(goal_vec)
        if goal_dist < 1:
            return 8  # Stay if already at goal

        goal_dir = goal_vec / goal_dist

        best_score = -np.inf
        best_action = 0

        for idx, (dx, dy) in enumerate(ACTIONS[:-1]):  # exclude 'stay'
            move_vec = np.array([dx, dy])
            move_norm = np.linalg.norm(move_vec)
            if move_norm == 0:
                continue

            move_dir = move_vec / move_norm
            angle_to_wind = np.arccos(np.clip(np.dot(move_dir, wind_dir), -1, 1))
            angle_deg = np.degrees(angle_to_wind)

            if angle_deg < 45:
                continue  # too close to the wind ("no-go zone")

            # Efficiency: best at 90°, worse at 45° or 135°
            sailing_efficiency = np.cos(np.radians(abs(angle_deg - 90)))
            goal_alignment = np.dot(move_dir, goal_dir)

            score = 0.6 * goal_alignment + 0.4 * sailing_efficiency

            if score > best_score:
                best_score = score
                best_action = idx

        return best_action

    def reset(self) -> None:
        pass

    def seed(self, seed: int = None) -> None:
        self.np_random = np.random.default_rng(seed)

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
