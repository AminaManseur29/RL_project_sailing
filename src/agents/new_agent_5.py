import numpy as np  # type: ignore
from agents.base_agent import BaseAgent


class WindAwareAgent(BaseAgent):
    """
    A clean rule-based sailing agent:
    Moves towards the goal while avoiding sailing directly into the wind.
    """

    def __init__(self):
        super().__init__()
        self.goal = np.array([16, 31])  # goal at top center (32x32 grid)

        # Unit direction vectors: N, NE, E, SE, S, SW, W, NW, Stay
        self.directions = np.array([
            [0, 1], [1, 1], [1, 0], [1, -1],
            [0, -1], [-1, -1], [-1, 0], [-1, 1],
            [0, 0],
        ])

    @staticmethod
    def angle_between(v1, v2):
        """Compute angle between two vectors in degrees."""
        v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
        dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        return np.degrees(np.arccos(dot_product))

    def act(self, observation):
        """
        Decide the next move.

        Args:
            observation: [x, y, vx, vy, wx, wy]
        Returns:
            action index [0–8]
        """
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]
        pos = np.array([x, y])
        wind = np.array([wx, wy])
        current_distance = np.linalg.norm(self.goal - pos)

        best_score = -np.inf
        best_action = 8  # default: stay

        for idx, direction in enumerate(self.directions):
            if np.all(direction == 0):
                continue  # skip staying; only pick if no better move

            wind_angle = self.angle_between(direction, -wind)

            if wind_angle < 45:  # no-go zone: upwind
                continue

            new_pos = pos + direction
            new_distance = np.linalg.norm(self.goal - new_pos)
            score = current_distance - new_distance

            if score > best_score:
                best_score = score
                best_action = idx

        return best_action

    def reset(self):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)

    def save(self, path):
        pass

    def load(self, path):
        pass
