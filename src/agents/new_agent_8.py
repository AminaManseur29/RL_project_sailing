"""
Agent avoiding no-go zone, maximizing progress,
and favoring directions roughly perpendicular to wind.
"""

import numpy as np
from agents.base_agent import BaseAgent


class OrthogonalWindAgent(BaseAgent):
    """
    Agent that avoids no-go zone (sailing directly upwind),
    tries to maximize progress toward the goal,
    and favors directions roughly perpendicular to the wind,
    which is generally the most efficient sailing angle.
    """

    def __init__(self):
        super().__init__()
        # Goal position set to top center of the grid
        self.goal = np.array([16, 31])

        # Possible movement directions (8 compass + stay still)
        self.directions = np.array([
            [0, 1],    # North
            [1, 1],    # Northeast
            [1, 0],    # East
            [1, -1],   # Southeast
            [0, -1],   # South
            [-1, -1],  # Southwest
            [-1, 0],   # West
            [-1, 1],   # Northwest
            [0, 0],    # Stay in place
        ])

    def angle_between(self, v1, v2):
        """
        Compute the angle in degrees between two vectors v1 and v2.
        """
        # Normalize vectors; add small epsilon to avoid division by zero
        v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
        # Clip dot product to avoid numerical errors outside valid arccos range [-1,1]
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        # Return angle in degrees
        return np.degrees(np.arccos(dot))

    def act(self, observation):
        """
        Decide the next action based on the current observation.

        observation: array with at least 6 elements:
            [x, y, vx, vy, wx, wy]
            - x, y: current position
            - wx, wy: wind vector components

        Returns:
            int: index of the chosen direction in self.directions
        """
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]

        pos = np.array([x, y])          # current position
        wind = np.array([wx, wy])       # wind vector
        to_goal = self.goal - pos       # vector from current position to goal
        current_dist = np.linalg.norm(to_goal)  # distance to goal

        best_score = -np.inf
        best_action = 8  # default: stay

        for idx, direction in enumerate(self.directions):
            if np.all(direction == 0):
                # Skip directions inside no-go zone (too close to headwind)
                continue

            # Calculate progress toward goal if moving in this direction
            wind_angle = self.angle_between(direction, -wind)
            if wind_angle < 45:
                continue  # skip: upwind

            # Sailing efficiency favors directions near 90 degrees to wind
            next_pos = pos + direction
            next_dist = np.linalg.norm(self.goal - next_pos)
            progress = current_dist - next_dist

            # Sailing efficiency: prefer ~90° to wind
            sail_angle = abs(90 - self.angle_between(direction, wind))
            # Gaussian weighting centered at 90° with std deviation ~45°
            efficiency = np.exp(- (sail_angle / 45)**2)  # Gaussian around 90°

            # Combine progress and sailing efficiency into score
            score = progress + 0.5 * efficiency

            # Keep track of best scoring action
            if score > best_score:
                best_score = score
                best_action = idx

        return best_action

    def reset(self):
        # No internal state to reset
        pass

    def seed(self, seed=None):
        # Set the numpy random seed for reproducibility if needed
        np.random.seed(seed)

    def save(self, path):
        # No model parameters to save
        pass

    def load(self, path):
        # No model parameters to load
        pass
