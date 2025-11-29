from typing import Any

import numpy as np
import pyenki

from ... import EPuckConfig, ParallelEnkiEnv, ThymioConfig
from ..same_robots.environment import facing_each_other, reward


def scenario(seed: int, copy_rng_from: pyenki.World | None = None) -> pyenki.World:
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
    epuck = pyenki.EPuck()
    epuck.position = (15, 0)
    epuck.angle = rng.uniform(0, 2 * np.pi)
    world.add_object(epuck)
    return world


def make_env(**kwargs: Any) -> ParallelEnkiEnv:
    thymio_config = ThymioConfig(reward=reward,
                                 terminations=[facing_each_other])
    thymio_config.action.fix_position = True
    thymio_config.action.dtype = np.float32
    thymio_config.observation.dtype = np.float32
    epuck_config = EPuckConfig(reward=reward,
                               terminations=[facing_each_other])
    epuck_config.action.fix_position = True
    epuck_config.action.dtype = np.float32
    epuck_config.observation.dtype = np.float32
    config = {'thymio': thymio_config, 'e-puck': epuck_config}
    env = ParallelEnkiEnv(scenario=scenario,
                          config=config,
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
