from __future__ import annotations

import sys
from typing import Any

import gymnasium as gym
import numpy as np
import pyenki

from ...types import Action, EpisodeStart, Observation, State
from .environment import make_env


class Baseline:

    @property
    def action_space(self) -> gym.Space[Any]:
        return gym.spaces.Box(0, 1, (1, ), dtype=np.float64)

    @property
    def observation_space(self) -> gym.Space[Any]:
        return gym.spaces.Dict(
            {'prox/value': gym.spaces.Box(0, 1, (7, ), dtype=np.float64)})

    def predict(self,
                observation: Observation,
                state: State | None = None,
                episode_start: EpisodeStart | None = None,
                deterministic: bool = False) -> tuple[Action, State | None]:
        prox = observation['prox/value']
        if any(prox > 0):
            prox = prox / np.max(prox)
            ws = np.array((0.5, 0.25, 0, -0.25, -0.5, 1, 1))
            w = np.dot(ws, prox)
        else:
            w = 1
        return np.clip([w], -1, 1), None


if __name__ == '__main__':
    display = '--display' in sys.argv
    if display:
        import pyenki.viewer
        pyenki.viewer.init()

    env = make_env(render_mode="human" if display else None)
    policy = Baseline()
    for i in range(10):
        data = env.rollout({'': policy}, seed=i)['thymio']
        print(f'episode {i}: reward={data.episode_reward:.1f}, steps={data.episode_length}')
    if display:
        pyenki.viewer.cleanup()
