import numpy as np
from agents.base_agent import BaseAgent


class SailingSmartAgent(BaseAgent):
    def __init__(self, goal_x=17, goal_y=31):
        super().__init__()
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.np_random = np.random.default_rng()

    def reset(self) -> None:
        pass

    def seed(self, seed: int = None) -> None:
        self.np_random = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        # Extract boat position, velocity, local wind
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        # Compute vectors
        # boat_pos = np.array([x, y])
        boat_vel = np.array([vx, vy])
        wind_vector = np.array([wx, wy])
        goal_vector = np.array([self.goal_x - x, self.goal_y - y])

        if np.linalg.norm(goal_vector) == 0:
            return 8  # Already at goal, stay in place

        goal_dir = goal_vector / np.linalg.norm(goal_vector)
        wind_dir = wind_vector / (np.linalg.norm(wind_vector) + 1e-6)

        # Define action directions (unit vectors)
        directions = np.array([
            [0, 1],    # 0: North
            [1, 1],    # 1: NE
            [1, 0],    # 2: E
            [1, -1],   # 3: SE
            [0, -1],   # 4: S
            [-1, -1],  # 5: SW
            [-1, 0],   # 6: W
            [-1, 1],   # 7: NW
            [0, 0],    # 8: Stay
        ])
        directions = np.array([d / (np.linalg.norm(d) + 1e-6) for d in directions])

        # Score each action
        best_score = -np.inf
        best_action = 0
        for i, action_dir in enumerate(directions):
            if i == 8:
                continue  # Skip "stay"

            # Alignment with goal
            goal_alignment = np.dot(action_dir, goal_dir)

            # Angle to wind
            wind_alignment = np.dot(action_dir, wind_dir)
            wind_angle = np.arccos(np.clip(wind_alignment, -1, 1)) * 180 / np.pi

            # Sailing efficiency score
            if wind_angle < 45:
                efficiency = 0  # No-go zone
            elif 45 <= wind_angle < 90:
                efficiency = 0.7  # Close-hauled
            elif 90 <= wind_angle < 135:
                efficiency = 1.0  # Beam reach (best)
            else:
                efficiency = 0.8  # Broad reach / running

            # Bonus if keeping similar direction to current velocity (smooth turns)
            if np.linalg.norm(boat_vel) > 0.1:
                vel_dir = boat_vel / np.linalg.norm(boat_vel)
                smoothness = np.dot(action_dir, vel_dir)
            else:
                smoothness = 1.0  # No speed → any direction is fine

            # Combined score
            score = goal_alignment * efficiency * smoothness

            if score > best_score:
                best_score = score
                best_action = i

        return best_action

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
