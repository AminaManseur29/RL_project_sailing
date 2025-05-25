"""
Sailing agent that avoids no-go zones,
optimizes angle toward goal,
and uses randomness to break ties.
"""

import numpy as np
from agents.base_agent import BaseAgent


class WindAwareSmartAgent(BaseAgent):
    """
    Sailing agent using soft randomness and angle optimization,
    avoiding no-go zones and seeking the most efficient path.
    """

    def __init__(self):
        super().__init__()
        # Goal position: top center on a 32x32 grid
        self.goal = np.array([16, 31])
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

        # Random number generator for stochastic tie breaking
        self.rng = np.random.default_rng()

    def angle_between(self, v1, v2):
        """
        Compute the angle in degrees between two vectors v1 and v2.
        """
        # Normalize vectors, add epsilon to avoid division by zero
        v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
        # Clip dot product to avoid numerical errors outside [-1,1]
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        # Return angle in degrees
        return np.degrees(np.arccos(dot))

    def act(self, observation):
        """
        Select an action given the current observation.

        observation: numpy array with at least 6 elements:
            [x, y, vx, vy, wx, wy]
            - x, y: current position
            - wx, wy: wind vector components

        Returns:
            int: index of action chosen from self.directions
        """
        x, y = observation[0], observation[1]
        wx, wy = observation[4], observation[5]

        pos = np.array([x, y])          # current position
        wind = np.array([wx, wy])       # wind vector
        to_goal = self.goal - pos       # vector pointing from current position to goal
        current_dist = np.linalg.norm(to_goal)  # distance to goal

        best_score = -np.inf            # best score found so far
        best_actions = []               # list of actions tied for best score

        for idx, direction in enumerate(self.directions):
            if np.all(direction == 0):
                continue  # skip staying still

            # Calculate angle between direction and headwind (-wind)
            wind_angle = self.angle_between(direction, -wind)
            if wind_angle < 45:
                continue  # Skip no-go

            # Calculate alignment of the chosen direction with the goal vector
            goal_angle = self.angle_between(direction, to_goal)
            # Convert angle to cosine similarity: 1 = perfect alignment, -1 = opposite
            goal_alignment = np.cos(np.radians(goal_angle))

            # Calculate improvement in distance to the goal by moving in this direction
            next_pos = pos + direction
            next_dist = np.linalg.norm(self.goal - next_pos)
            progress = current_dist - next_dist

            # Combine progress and goal alignment to get a composite score
            # Weigh progress more, but add half weight to alignment
            score = progress + 0.5 * goal_alignment

            # Track the best scoring actions
            if score > best_score:
                best_score = score
                best_actions = [idx]
            elif np.isclose(score, best_score, atol=1e-3):
                best_actions.append(idx)

        # If there are multiple equally good actions, pick one randomly
        if best_actions:
            return int(self.rng.choice(best_actions))
        else:
            return 8  # stay in place if no valid move

    def reset(self):
        # No internal state to reset for this agent
        pass

    def seed(self, seed=None):
        # Reset RNG for reproducibility if needed
        self.rng = np.random.default_rng(seed)

    def save(self, path):
        # No model parameters to save
        pass

    def load(self, path):
        # No model parameters to load
        pass
