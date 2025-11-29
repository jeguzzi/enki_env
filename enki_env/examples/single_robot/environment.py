from __future__ import annotations

from typing import Any

import gymnasium as gym
import pyenki

from ... import BaseEnv, ThymioConfig
from ...types import Termination
from ..utils import is_still, normalize_angle


def scenario(seed: int,
             copy_rng_from: pyenki.World | None = None) -> pyenki.World:
    world = pyenki.World(seed=seed)
    if copy_rng_from:
        world.copy_random_generator(copy_rng_from)
    rng = world.random_generator
    robot = pyenki.Thymio2()
    robot.position = (-18, 0)
    robot.angle = rng.uniform(-1.2, 1.2)
    rgb = rng.uniform(0.1, 1, size=3)
    rgb /= max(rgb)
    robot.set_led_top(*rgb)
    world.add_object(robot)
    wall = pyenki.PhysicalObject(lx=10,
                                 ly=100,
                                 height=10,
                                 mass=-1,
                                 color=pyenki.Color.darkred)
    world.add_object(wall)
    return world


def is_standing_in_front_of_wall(angle_tol: float = 0.05,
                                 speed_tol: float = 1) -> Termination:

    def f(robot: pyenki.DifferentialWheeled) -> bool | None:
        if abs(normalize_angle(robot.angle)) < angle_tol and is_still(
                robot, speed_tol):
            # Success
            return True
        return None

    return f


def reward(robot: pyenki.DifferentialWheeled, success: bool | None) -> float:
    return -1 - abs(normalize_angle(robot.angle)) - 0.1 * (abs(
        robot.left_wheel_encoder_speed) + abs(robot.right_wheel_encoder_speed))


def make_env(**kwargs: Any) -> BaseEnv:
    config = ThymioConfig(reward=reward,
                          terminations=[
                              is_standing_in_front_of_wall(angle_tol=0.05,
                                                           speed_tol=1)
                          ])
    config.action.fix_position = True
    config.observation.speed = True
    env = gym.make("Enki",
                   max_duration=2,
                   scenario=scenario,
                   config=config,
                   default_success=False,
                   render_kwargs=dict(camera_pitch=-1.57,
                                      camera_position=(-10, 0),
                                      camera_altitude=60),
                   **kwargs)
    return env


if __name__ == '__main__':
    env = make_env()
    print(f'Action space: {env.action_space}')
    print(f'Observation space: {env.observation_space}')
    obs, info = env.reset()
    print(f'First observation: {obs}')
