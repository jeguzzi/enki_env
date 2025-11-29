from __future__ import annotations

from typing import Any

import numpy as np
import pyenki

from ... import ParallelEnkiEnv, ThymioConfig
from ..utils import is_still, normalize_angle


def scenario(seed: int,
             copy_rng_from: pyenki.World | None = None) -> pyenki.World:
    world = pyenki.World(seed=seed)
    if copy_rng_from:
        world.copy_random_generator(copy_rng_from)
    rng = world.random_generator
    rgb = rng.uniform(0.1, 1, size=3)
    rgb /= max(rgb)
    robot = pyenki.Thymio2()
    robot.angle = rng.uniform(0, 2 * np.pi)
    robot.set_led_top(*rgb)
    world.add_object(robot)
    robot = pyenki.Thymio2()
    robot.position = (20, 0)
    robot.angle = rng.uniform(0, 2 * np.pi)
    robot.set_led_top(*rgb)
    world.add_object(robot)
    return world


def is_facing(robot: pyenki.Robot,
              other: pyenki.Robot,
              tol: float = 0.05) -> bool:
    delta = other.position - robot.position
    angle = np.arctan2(delta[1], delta[0])
    return abs(normalize_angle(robot.angle - angle)) < tol


def is_standing_in_front_of_other_robot(robot: pyenki.DifferentialWheeled,
                                        angle_tol: float = 0.05,
                                        speed_tol: float = 1) -> bool | None:
    other = [r for r in robot.world.robots if r is not robot][0]
    if is_facing(robot, other, angle_tol) and is_still(robot, speed_tol):
        return True
    return None


def reward(robot: pyenki.DifferentialWheeled, success: bool | None) -> float:
    other = [r for r in robot.world.robots if r is not robot][0]
    delta = other.position - robot.position
    angle = np.arctan2(delta[1], delta[0])
    d = abs(normalize_angle(robot.angle - angle))
    w = max(0, 0.5 - d) * 0.2
    # w = 0.1
    speeds = abs(robot.left_wheel_encoder_speed) + abs(
        robot.right_wheel_encoder_speed)
    return (0 if success else -1) - d - w * speeds


def make_env(**kwargs: Any) -> ParallelEnkiEnv:
    config = ThymioConfig(reward=reward,
                          terminations=[is_standing_in_front_of_other_robot])
    config.action.fix_position = True
    config.observation.speed = True
    env = ParallelEnkiEnv(scenario=scenario,
                          config={'thymio': config},
                          max_duration=5,
                          default_success=False,
                          render_kwargs=dict(camera_pitch=-1.57,
                                             camera_position=(10, 0),
                                             camera_altitude=60),
                          **kwargs)
    return env


if __name__ == '__main__':
    env = make_env()
    print(f'Action space: {env.action_spaces}')
    print(f'Observation space: {env.observation_spaces}')
    obs, info = env.reset()
    print(f'First observation: {obs}')
