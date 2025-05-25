"""
Wind-Aware Agent for the Sailing Challenge

This file contains a simple rule-based agent that moves in the direction of the goal, 
while avoiding sailing into the wind.
"""

import numpy as np  # type: ignore
from agents.base_agent import BaseAgent


class WindAwareAgentWithEfficiency(BaseAgent):
    """
    Improved sailing agent that moves toward the goal
    while avoiding the no-go zone and considering sailing efficiency.
    """

    def __init__(self):
        """Initialize the agent."""
        super().__init__()
        self.np_random = np.random.default_rng()
        self.goal = np.array([16, 31])  # goal is at the top center of 32x32 grid

        # Unit direction vectors
        self.directions = np.array([
            [0, 1],    # 0: N
            [1, 1],    # 1: NE
            [1, 0],    # 2: E
            [1, -1],   # 3: SE
            [0, -1],   # 4: S
            [-1, -1],  # 5: SW
            [-1, 0],   # 6: W
            [-1, 1],   # 7: NW
            [0, 0],    # 8: Stay in place
        ])

    def angle_between(self, v1: np.ndarray, v2: np.ndarray):
        """
        Compute angle between two vectors (in degrees).
        """
        v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle = np.arccos(dot)
        return np.degrees(angle)

    def sailing_efficiency(self, angle_to_wind: float) -> float:
        """
        Estimate sailing efficiency based on angle to wind.
        """
        if angle_to_wind < 40:
            return 0.0  # no-go zone, unusable
        elif 40 <= angle_to_wind < 60:
            return 0.6  # close-hauled, decent
        elif 60 <= angle_to_wind < 120:
            return 1.0  # beam reach, best
        elif 120 <= angle_to_wind < 150:
            return 0.7  # broad reach, good
        else:
            return 0.4  # running, less efficient

    def act(self, observation: np.ndarray) -> int:
        """
        Choose direction toward the goal, factoring in sailing efficiency.
        """
        x, y, vx, vy, wx, wy = observation[:6]
        pos = np.array([x, y])
        wind_dir = np.array([wx, wy])

        curr_dist = np.linalg.norm(self.goal - pos)
        scores = []

        for direction in self.directions:
            if np.all(direction == 0):
                scores.append(-np.inf)
                continue  # skip staying in place

            wind_angle = self.angle_between(direction, -wind_dir)
            efficiency = self.sailing_efficiency(wind_angle)

            if efficiency == 0.0:
                scores.append(-np.inf)
                continue  # skip no-go directions

            # if wind_angle < 45:  # No-go zone: upwind
            #     scores.append(-np.inf)
            #     continue

            # Goal progress (distance reduction)
            next_pos = pos + direction
            next_dist = np.linalg.norm(self.goal - next_pos)
            progress = curr_dist - next_dist

            goal_progress = progress * efficiency
            scores.append(goal_progress)

        # Choose best scoring direction
        best_action = int(np.argmax(scores))
        return best_action

    def reset(self) -> None:
        """Reset the agent's internal state."""
        pass

    def seed(self, seed: int = None) -> None:
        """Set the random seed."""
        self.np_random = np.random.default_rng(seed)

    def save(self, path: str) -> None:
        """Save the agent's parameters (none in this simple agent)."""
        pass

    def load(self, path: str) -> None:
        """Load the agent's parameters (none in this simple agent)."""
        # No parameters to load for this simple agent
        pass
