import numpy as np
from agents.base_agent import BaseAgent

actions = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0)
]

position = [17, 30]


class SmartSailingAgent(BaseAgent):
    def __init__(
        self, action_space=actions, goal_x=position[0], goal_y=position[1], grid_size=[32, 32]
    ):
        self.actions = action_space  # list of (dx, dy)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.grid_size = grid_size
        self.current_tack = None  # 'left' or 'right'
        self.tack_duration = 0

    def reset(self) -> None:
        pass

    def seed(self, seed: int = None) -> None:
        self.np_random = np.random.default_rng(seed)

    def decode_wind_field(self, observation):
        grid_x, grid_y = self.grid_size
        wind_data = observation[6:]
        wind_field = np.zeros((grid_x, grid_y, 2))
        idx = 0
        for i in range(grid_x):
            for j in range(grid_y):
                wind_field[i, j, 0] = wind_data[idx]
                wind_field[i, j, 1] = wind_data[idx + 1]
                idx += 2
        return wind_field

    def act(self, observation):
        x, y = int(observation[0]), int(observation[1])
        # vx, vy = observation[2], observation[3]
        wx, wy = observation[4], observation[5]

        goal_vector = np.array([self.goal_x - x, self.goal_y - y])
        goal_dir = goal_vector / (np.linalg.norm(goal_vector) + 1e-8)

        wind_vector = np.array([wx, wy])
        wind_dir = wind_vector / (np.linalg.norm(wind_vector) + 1e-8)

        wind_field = self.decode_wind_field(observation)

        best_score = -np.inf
        best_action = 0

        for i, (dx, dy) in enumerate(self.actions):
            if dx == 0 and dy == 0:
                continue  # skip stay in place

            action_vector = np.array([dx, dy])
            action_dir = action_vector / (np.linalg.norm(action_vector) + 1e-8)

            # Alignment to goal
            goal_alignment = np.dot(action_dir, goal_dir)

            # Angle to wind
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
                efficiency = 0.4  # running

            # Look ahead: estimate wind in next cell
            next_x, next_y = np.clip(x + dx, 0, self.grid_size[0] - 1), np.clip(y + dy, 0, self.grid_size[1] - 1)
            next_wind = wind_field[next_x, next_y]
            next_wind_strength = np.linalg.norm(next_wind)

            # Favor directions with good future wind
            wind_bonus = next_wind_strength

            # Tack control: if upwind, stick to current tack
            if angle_to_wind < 60:
                if self.current_tack is None:
                    self.current_tack = 'left' if np.cross(wind_dir, goal_dir) > 0 else 'right'
                    self.tack_duration = 0

                tack_direction = np.cross(wind_dir, action_dir)
                if self.current_tack == 'left' and tack_direction < 0:
                    tack_penalty = -0.5
                elif self.current_tack == 'right' and tack_direction > 0:
                    tack_penalty = -0.5
                else:
                    tack_penalty = 0

                self.tack_duration += 1
                if self.tack_duration > 20:
                    self.current_tack = 'left' if self.current_tack == 'right' else 'right'
                    self.tack_duration = 0
            else:
                tack_penalty = 0

            # Final score
            score = goal_alignment * efficiency + 0.1 * wind_bonus + tack_penalty

            if score > best_score:
                best_score = score
                best_action = i

        return best_action

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
