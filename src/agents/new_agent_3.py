import numpy as np
from agents.base_agent import BaseAgent

actions = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0)
]

position = [17, 30]


class AdvancedSailingAgent(BaseAgent):
    def __init__(self, goal=position, grid_size=[32, 32]):
        self.goal = np.array(goal)
        self.grid_size = grid_size
        self.actions = np.array([
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1),
            (0, 0)
        ])

    def reset(self) -> None:
        pass

    def seed(self, seed: int = None) -> None:
        self.np_random = np.random.default_rng(seed)

    def angle_between(self, v1, v2):
        unit_v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        unit_v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        dot = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        return np.arccos(dot)  # in radians
    
    def sailing_efficiency(self, action_vector, wind_vector):
        angle = np.degrees(self.angle_between(action_vector, wind_vector))
        angle = angle % 360

        if angle < 45 or angle > 315:
            return 0.1  # no-go zone
        elif 45 <= angle < 60 or 300 <= angle <= 315:
            return 0.5  # close-hauled
        elif 60 <= angle < 120 or 240 <= angle <= 300:
            return 1.0  # beam reach (optimal)
        elif 120 <= angle < 150 or 210 <= angle < 240:
            return 0.7  # broad reach
        else:
            return 0.4  # running (wind from behind)

    def act(self, observation):
        position = observation[0:2]
        velocity = observation[2:4]
        wind_local = observation[4:6]

        to_goal = self.goal - position
        to_goal_norm = to_goal / (np.linalg.norm(to_goal) + 1e-8)

        best_score = -np.inf
        best_action = 8  # stay in place

        for i, action in enumerate(self.actions):
            action_vector = np.array(action)
            if np.linalg.norm(action_vector) == 0:
                continue  # skip staying for now

            # Align with goal
            goal_alignment = np.dot(action_vector, to_goal_norm)

            # Sailing efficiency w.r.t. wind
            sail_eff = self.sailing_efficiency(action_vector, wind_local)

            # Maintain inertia (avoid sharp turns)
            inertia_alignment = np.dot(action_vector, velocity / (np.linalg.norm(velocity) + 1e-8))

            # Combine into a score
            score = (2.0 * goal_alignment) + (3.0 * sail_eff) + (1.0 * inertia_alignment)

            if score > best_score:
                best_score = score
                best_action = i

        return best_action

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
