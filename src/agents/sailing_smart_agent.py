"""
Naive Agent for the Sailing Challenge

This file provides a simple agent that always goes north.
Students can use it as a reference for how to implement a sailing agent.
"""


import numpy as np                # type: ignore
from agents.base_agent import BaseAgent


class SailingSmartAgent(BaseAgent):
    """
    A smart agent for the Sailing Challenge.
    """

    def __init__(self, alpha=0.6, beta=0.4, goal=[17, 31]):
        """Initialize the agent."""
        super().__init__()
        self.np_random = np.random.default_rng()
        self.alpha = alpha  # target weight
        self.beta = beta    # sailing efficiency weight
        self.goal = goal
        self.action_directions = np.array([
            [0, 1],    # 0: North
            [1, 1],    # 1: Northeast
            [1, 0],    # 2: East
            [1, -1],   # 3: Southeast
            [0, -1],   # 4: South
            [-1, -1],  # 5: Southwest
            [-1, 0],   # 6: West
            [-1, 1],   # 7: Northwest
            [0, 0],    # 8: Stay
        ])

    def act(self, observation: np.ndarray) -> int:
        """
        Select an action based on the current observation.

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
        x, y = observation[0], observation[1]
        # vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        goal_x, goal_y = self.goal

        # Compute target direction
        target_vector = np.array([goal_x - x, goal_y - y])
        target_norm = np.linalg.norm(target_vector)
        target_dir = target_vector / target_norm if target_norm > 0 else np.array([0, 0])

        # Compute wind direction
        wind_vector = np.array([wx, wy])
        wind_norm = np.linalg.norm(wind_vector)
        wind_dir = wind_vector / wind_norm if wind_norm > 0 else np.array([0, 0])

        # Initialize scores
        scores = np.full(9, -np.inf)

        for i, action_vec in enumerate(self.action_directions):
            if np.all(action_vec == 0):
                continue  # skip "stay" unless no other option

            action_dir = action_vec / np.linalg.norm(action_vec)

            # Angle between action and wind (in degrees)
            cos_angle = np.clip(np.dot(action_dir, wind_dir), -1, 1)
            angle = np.degrees(np.arccos(cos_angle))

            # Sailing efficiency score
            if angle < 45:
                sail_score = -1.0  # no-go zone (forbidden)
            elif angle < 60:
                sail_score = 0.2   # close-hauled
            elif angle < 105:
                sail_score = 1.0   # beam reach (optimal)
            elif angle < 150:
                sail_score = 0.8   # broad reach (very good)
            else:
                sail_score = 0.5   # running (acceptable)

            if sail_score < 0:
                continue  # skip no-go actions

            # Target alignment score
            target_alignment = np.dot(action_dir, target_dir)

            # Combined score
            combined_score = self.alpha * target_alignment + self.beta * sail_score
            scores[i] = combined_score

        # Choose best action (fallback: stay if all others forbidden)
        if np.all(scores == -np.inf):
            best_action = 8  # stay
        else:
            best_action = np.argmax(scores)

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
