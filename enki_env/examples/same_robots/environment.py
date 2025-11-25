from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pyenki

from ... import ParallelEnkiEnv, ThymioAction, ThymioConfig
from ...config import RewardConfig


def scenario(seed: int) -> pyenki.World:
    world = pyenki.World(seed=seed)
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


def normalize_angle(angle: float) -> float:
    return math.fmod(angle + math.pi, 2 * math.pi) - math.pi


def is_facing(robot: pyenki.Robot,
              other: pyenki.Robot,
              tol: float = 0.1) -> bool:
    delta = other.position - robot.position
    angle = np.arctan2(delta[1], delta[0])
    return abs(normalize_angle(robot.angle - angle)) < tol


def facing_each_other(robot: pyenki.DifferentialWheeled,
                      world: pyenki.World) -> bool | None:
    r1, r2 = world.robots
    if is_facing(r1, r2) and is_facing(r2, r1):
        return True
    return None


class Reward(RewardConfig):

    def get(self, robot: pyenki.DifferentialWheeled,
            world: pyenki.World) -> float:
        other = [r for r in world.robots if r is not robot][0]
        delta = other.position - robot.position
        angle = np.arctan2(delta[1], delta[0])
        return -1 - abs(normalize_angle(robot.angle - angle))


def make_env(**kwargs: Any) -> ParallelEnkiEnv:
    config = ThymioConfig(reward=Reward(), terminations=[facing_each_other])
    cast('ThymioAction', config.action).fix_position = True
    env = ParallelEnkiEnv(scenario=scenario,
                          config={'': config},
                          max_duration=5,
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
