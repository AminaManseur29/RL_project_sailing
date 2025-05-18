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

