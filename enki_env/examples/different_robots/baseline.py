import sys
from typing import Any

import gymnasium as gym
import numpy as np
import pyenki

from ...types import Action, EpisodeStart, Observation, Predictor, State
from ..same_robots.baseline import Baseline as ThymioBaseline
from .environment import make_env


class EPuckBaseline:

    @property
    def action_space(self) -> gym.Space[Any]:
        return gym.spaces.Box(0, 1, (1, ), dtype=np.float32)

    @property
    def observation_space(self) -> gym.Space[Any]:
        return gym.spaces.Dict(
            {'prox/value': gym.spaces.Box(0, 1, (8, ), dtype=np.float32)})

    def predict(self,
                observation: Observation,
                state: State | None = None,
                episode_start: EpisodeStart | None = None,
                deterministic: bool = False) -> tuple[Action, State | None]:
        prox = np.atleast_2d(observation['prox/value'])
        m = np.max(prox, axis=-1)
        prox[m > 0] /= m[:, np.newaxis][m > 0]
        ws = np.array([(-0.1, -0.25, -0.5, -1, -1, 0.5, 0.25, 0.1)], dtype=np.float32)
        w = np.tensordot(prox, ws, axes=([1], [1]))
        w[m == 0] = 1
        return np.clip(w, -1, 1), None


if __name__ == '__main__':
    display = '--display' in sys.argv
    if display:
        import pyenki.viewer
        pyenki.viewer.init()

    env = make_env(render_mode="human" if display else None)
    policies: dict[str, Predictor] = {'thymio': ThymioBaseline(), 'e-puck': EPuckBaseline()}
    for i in range(10):
        rs = env.rollout(policies, seed=i)
        print(f'episode {i}:')
        for group, data in rs.items():
            print(
                f'  -{group}: reward={data.episode_reward:.1f}, '
                f'steps={data.episode_length}, '
                f'success={data.episode_success[0] if data.episode_success is not None else "?"}'
            )
    if display:
        pyenki.viewer.cleanup()
