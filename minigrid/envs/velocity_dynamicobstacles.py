from __future__ import annotations

from operator import add

from gymnasium.spaces import Discrete

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Goal
from minigrid.minigrid_env import MiniGridEnv


class VelocityDynamicObstaclesEnv(MiniGridEnv):
    """
    ## Description

    This environment is an empty room with moving obstacles.
    The goal of the agent is to reach the green goal square without colliding
    with any obstacle. A large penalty is subtracted if the agent collides with
    an obstacle and the episode finishes. This environment is useful to test
    Dynamic Obstacle Avoidance for mobile robots with Reinforcement Learning in
    Partial Observability.

    ## Mission Space

    "get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure. A '-1' penalty is
    subtracted if the agent collides with an obstacle.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent collides with an obstacle.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Dynamic-Obstacles-5x5-v0`
    - `MiniGrid-Dynamic-Obstacles-Random-5x5-v0`
    - `MiniGrid-Dynamic-Obstacles-6x6-v0`
    - `MiniGrid-Dynamic-Obstacles-Random-6x6-v0`
    - `MiniGrid-Dynamic-Obstacles-8x8-v0`
    - `MiniGrid-Dynamic-Obstacles-16x16-v0`

    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        n_obstacles=4,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        # Ensure one obstacle per row
        self.n_obstacles = min(n_obstacles - 2, size)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.action_space = Discrete(self.actions.forward + 1)
        self.reward_range = (-1, 1)

        # Initialize velocities for obstacles
        self.obstacle_velocities = []
        self.obstacle_colors = []
        velocity_color_map = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'red'}
        for i in range(self.n_obstacles):
            # Randomly choose velocity between 0 and 3 inclusive for x direction
            velocity_x = self._rand_int(1, 4)  # 0 to 3 inclusive
            # velocity_x = self._rand_int(0, 4)  # 0 to 3 inclusive
            # Randomly choose direction: -1 for left, +1 for right
            direction = self._rand_int(0, 2) * 2 - 1  # converts 0,1 to -1,1
            self.obstacle_velocities.append((velocity_x * direction, 0))
            self.obstacle_colors.append(velocity_color_map[velocity_x])

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(width - 2, height - 2, Goal())

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.obstacles = []
        # Start placing obstacles from the second row and stop before the last row
        for i_obst in range(self.n_obstacles):
            # Assign color based on velocity
            color = self.obstacle_colors[i_obst]
            self.obstacles.append(Ball(color=color))
            # Randomize the starting column position for each obstacle
            random_col = self._rand_int(1, width - 1)
            self.place_obj(self.obstacles[i_obst], top=(random_col, i_obst + 2), size=(1, 1), max_tries=100)

        self.mission = "get to the green goal square"

    # NOTE: borders are occupied by the walls
    def step(self, action):
        if action >= self.action_space.n:
            action = 0

        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != "goal"
        
        # Process the player's action after updating obstacles
        obs, reward, terminated, truncated, info = super().step(action)

        # Update obstacle positions with horizontal movement simulation
        min_x = 1
        max_x = self.grid.width - 2

        for i_obst, obstacle in enumerate(self.obstacles):
            old_pos = obstacle.cur_pos
            vx, _ = self.obstacle_velocities[i_obst]

            path_positions = []
            x = old_pos[0]
            y = old_pos[1]
            vel = vx

            steps = abs(vel)
            step_dir = 1 if vel > 0 else -1 if vel < 0 else 0

            for _ in range(steps):
                if step_dir == 0:
                    break

                next_x = x + step_dir

                # Bounce off walls
                if next_x < min_x or next_x > max_x:
                    step_dir *= -1  # reverse direction
                    vel *= -1
                    next_x = x + step_dir  # move after bouncing

                x = next_x
                path_positions.append((x, y))

            # Update velocity after completing the move
            self.obstacle_velocities[i_obst] = (vel, 0)

            new_pos = (x, y)

            # Collision detection along the simulated path
            if any(pos == self.agent_pos for pos in path_positions):
                reward = -1
                terminated = True
                return self.gen_obs(), reward, terminated, False, {}

            # Move obstacle to the new position if the cell is empty
            if self.grid.get(*new_pos) is None:
                self.place_obj(obstacle, top=new_pos, size=(1, 1), max_tries=100)
                self.grid.set(old_pos[0], old_pos[1], None)

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            # Check if the agent is at the goal
            if front_cell and front_cell.type == "goal":
                reward = 1
                terminated = True
            else:
                reward = -1
                terminated = True
            return obs, reward, terminated, truncated, info

        return obs, reward, terminated, truncated, info

    def _get_path_positions(self, start_pos, end_pos):
        """Generate all positions along the path from start_pos to end_pos."""
        path_positions = []
        x0, y0 = start_pos
        x1, y1 = end_pos
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy))
        for step in range(1, steps + 1):
            x = x0 + (dx * step) // steps
            y = y0 + (dy * step) // steps
            path_positions.append((x, y))
        return path_positions
