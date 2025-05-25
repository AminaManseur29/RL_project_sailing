"""
Sailing agent that efficiently moves toward the goal,
avoiding the no-go zone (upwind) and maximizing progress.
"""

import numpy as np
from agents.base_agent import BaseAgent


class WindAwareEfficientAgent(BaseAgent):
    """
    Sailing agent that moves efficiently toward the goal,
    avoiding the no-go zone (upwind) and maximizing progress.
    """

    def __init__(self):
        super().__init__()
        # Goal position: top center of a 32x32 grid
        self.goal = np.array([16, 31])

        # Define 8 movement directions + stay
        self.directions = np.array([
            [0, 1],    # 0: North
            [1, 1],    # 1: Northeast
            [1, 0],    # 2: East
            [1, -1],   # 3: Southeast
            [0, -1],   # 4: South
            [-1, -1],  # 5: Southwest
            [-1, 0],   # 6: West
            [-1, 1],   # 7: Northwest
            [0, 0],    # 8: Stay in place
        ])

    def angle_between(self, v1, v2):
        """
        Compute the angle in degrees between two vectors v1 and v2.
        """
        # Normalize vectors (add small epsilon to avoid division by zero)
        v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
        # Dot product clipped to [-1, 1] to prevent numerical errors
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        # Return angle in degrees
        return np.degrees(np.arccos(dot))

    def sailing_efficiency(self, direction, wind):
        """
        Returns an efficiency factor [0, 1] indicating
        how well the chosen direction converts wind into progress.

        - 0 if inside the no-go zone (within 45 degrees facing the wind)
        - 0.3 if downwind (angle > 135 degrees)
        - 1 otherwise (crosswind, efficient sailing)
        """
        wind_angle = self.angle_between(direction, -wind)  # angle between direction and headwind
        if wind_angle < 45:
            return 0  # No-go zone: inefficient, cannot sail directly into the wind
        elif wind_angle > 135:
            return 0.3  # Downwind: less efficient sailing
        else:
            return 1  # Crosswind: efficient sailing

    def act(self, observation):
        """
        Given the current observation (position and wind),
        selects the best action to move toward the goal efficiently.

        observation: numpy array [x, y, vx, vy, wx, wy]
            x, y : current position
            wx, wy : wind components

        Returns:
            action (int): index in self.directions of the chosen move
        """
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]

        pos = np.array([x, y])      # current position
        wind = np.array([wx, wy])   # current wind vector
        current_dist = np.linalg.norm(self.goal - pos)   # distance to goal

        best_score = -np.inf    # best progress score found so far
        best_action = 8         # default action: stay in place

        for idx, direction in enumerate(self.directions):
            if np.all(direction == 0):
                continue  # Skip staying still for now

            next_pos = pos + direction              # position if this move is taken
            next_dist = np.linalg.norm(self.goal - next_pos)    # distance to goal after move
            progress = current_dist - next_dist     # positive if closer to goal

            efficiency = self.sailing_efficiency(direction, wind)   # sailing efficiency factor
            score = progress * efficiency                       # combined score

            if score > best_score:
                best_score = score
                best_action = idx   # update best action

        return best_action

    def reset(self):
        # No internal state to reset for this simple agent
        pass

    def seed(self, seed=None):
        # Set random seed for reproducibility if needed
        np.random.seed(seed)

    def save(self, path):
        # No parameters to save here
        pass

    def load(self, path):
        # No parameters to load here
        pass
