import random

import numpy as np


class Policy:
    def act(self, obs):
        """obs: dict from Rover.observation(). Returns (v in m/s, w in rad/s)."""
        raise NotImplementedError


class RandomWalkPolicy(Policy):
    """Forward exploration with LiDAR-based obstacle avoidance."""

    def __init__(self, hold_steps=40):
        self.hold = hold_steps
        self.count = 0
        self.cmd = (0.0, 0.0)
        self.rng = random.Random(42)

    def act(self, obs):
        ranges = np.asarray(obs.get("lidar"), dtype=float)
        if ranges.ndim == 2:
            scan = np.nanmin(ranges, axis=0)
        else:
            scan = ranges.reshape(-1)
        finite = np.where(np.isfinite(scan), scan, np.inf)
        width = len(finite)
        if width < 12:
            return 0.0, 0.0

        left = float(np.min(finite[: width // 3]))
        front = float(np.min(finite[width // 3: 2 * width // 3]))
        right = float(np.min(finite[2 * width // 3:]))
        front_left = float(np.min(finite[width // 3: width // 2]))
        front_right = float(np.min(finite[width // 2: 2 * width // 3]))

        stop_distance = 1.0
        caution_distance = 2.2
        if front < stop_distance:
            turn = 0.9 if front_left > front_right else -0.9
            return 0.0, turn

        speed = 0.32 if front >= caution_distance else 0.12
        steer = 0.0
        if front < caution_distance:
            steer = 0.7 if front_left > front_right else -0.7
        elif left < 1.2:
            steer = -0.35
        elif right < 1.2:
            steer = 0.35
        elif self.count <= 0:
            steer = self.rng.uniform(-0.15, 0.15)
            self.count = self.rng.randint(max(1, self.hold // 2), self.hold)
        else:
            steer = self.cmd[1]

        self.count -= 1
        self.cmd = speed, steer
        return self.cmd
