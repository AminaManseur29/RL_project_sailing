import numpy as np
from agents.base_agent import BaseAgent

actions = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0)
]

position = [17, 30]


class SmartSailingAgent(BaseAgent):
    def __init__(self, action_space=actions, goal_x=position[0], goal_y=position[1]):
        self.actions = action_space  # list of (dx, dy)
        self.goal_x = goal_x
        self.goal_y = goal_y

    def reset(self) -> None:
        pass

    def seed(self, seed: int = None) -> None:
        self.np_random = np.random.default_rng(seed)

    def act(self, observation):
        x, y = observation[0], observation[1]
        # vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        # Compute goal vector
        goal_vector = np.array([self.goal_x - x, self.goal_y - y])
        goal_dir = goal_vector / (np.linalg.norm(goal_vector) + 1e-8)

        # Compute wind vector
        wind_vector = np.array([wx, wy])
        wind_dir = wind_vector / (np.linalg.norm(wind_vector) + 1e-8)

        # For each action, score based on alignment to goal and efficient sailing angle
        best_score = -np.inf
        best_action = 0

        for i, (dx, dy) in enumerate(self.actions):
            if dx == 0 and dy == 0:
                continue  # skip "stay in place"

            action_vector = np.array([dx, dy])
            action_dir = action_vector / (np.linalg.norm(action_vector) + 1e-8)

            # Alignment to goal
            goal_alignment = np.dot(action_dir, goal_dir)

            # Angle to wind (cosine)
            wind_alignment = np.dot(action_dir, wind_dir)
            angle_to_wind = np.arccos(np.clip(wind_alignment, -1, 1)) * 180 / np.pi

            # Sailing efficiency score
            if angle_to_wind < 45:
                efficiency = 0.1  # no-go zone
            elif 45 <= angle_to_wind < 60:
                efficiency = 0.6  # close-hauled
            elif 60 <= angle_to_wind < 120:
                efficiency = 1.0  # beam reach
            elif 120 <= angle_to_wind < 150:
                efficiency = 0.7  # broad reach
            else:
                efficiency = 0.4  # running (wind from behind)

            # Final score: combine progress to goal + sailing efficiency
            score = goal_alignment * efficiency

            if score > best_score:
                best_score = score
                best_action = i

        return best_action

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
