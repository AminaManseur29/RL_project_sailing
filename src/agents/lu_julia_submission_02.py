"""
Wind-Aware Agent for the Sailing Challenge

This file contains a simple rule-based agent that moves in the direction of the goal, 
while avoiding sailing into the wind.
"""

import numpy as np  # type: ignore
from agents.base_agent import BaseAgent

class WindAwareAgent(BaseAgent):
    """
    A simple rule-based sailing agent that moves in the direction of the goal,
    while avoiding sailing into the wind.
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

    def act(self, observation: np.ndarray) -> int:
        """
        Choose the direction that brings the agent closest to the goal,
        while avoiding sailing into the wind.
        
        Args:
            observation: A numpy array containing the current observation.
                Format: [x, y, vx, vy, wx, wy] where:
                - (x, y) is the current position
                - (vx, vy) is the current velocity
                - (wx, wy) is the current wind vector
        
        Returns:
            action: An integer in [0, 8] representing the action to take:
                - 0: Move North
                - 1: Move Northeast
                - 2: Move East
                - 3: Move Southeast
                - 4: Move South
                - 5: Move Southwest
                - 6: Move West
                - 7: Move Northwest
                - 8: Stay in place
        """
        x, y, vx, vy, wx, wy = observation[:6]
        pos = np.array([x, y])
        wind_dir = np.array([wx, wy])

        curr_dist = np.linalg.norm(self.goal - pos)
        scores = []

        for direction in self.directions:
            wind_angle = self.angle_between(direction, -wind_dir)

            if wind_angle < 45:  # No-go zone: upwind
                scores.append(-np.inf)
                continue
            
            # Goal progress (distance reduction)
            next_pos = pos + direction
            next_dist = np.linalg.norm(self.goal - next_pos)
            goal_progress = curr_dist - next_dist
            scores.append(goal_progress)

        # Choose best scoring direction
        best_action = int(np.argmax(scores))
        return best_action

    def reset(self) -> None:
        """Reset the agent's internal state between episodes."""
        # Nothing to reset for this simple agent
        pass

    def seed(self, seed: int = None) -> None:
        """Set the random seed for reproducibility."""
        self.np_random = np.random.default_rng(seed)

    def save(self, path: str) -> None:
        """
        Save the agent's learned parameters to a file.
        
        Args:
            path: Path to save the agent's state
        """
        # No parameters to save for this simple agent
        pass

    def load(self, path: str) -> None:
        """
        Load the agent's learned parameters from a file.
        
        Args:
            path: Path to load the agent's state from
        """
        # No parameters to load for this simple agent
        pass
