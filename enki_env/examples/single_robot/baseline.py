from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import pyenki

from ... import EnkiEnv
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
        if any(prox[:5] > 0):
            w = 0.25 * (prox[0] + 2 * prox[1] - 2 * prox[3] - prox[4])
        else:
            w = 1
        return np.clip([w], -1, 1), None


if __name__ == '__main__':
    import pyenki.viewer

    pyenki.viewer.init()
    env = make_env(render_mode="human")
    policy = Baseline()
    for i in range(10):
        rew, steps = cast('EnkiEnv', env.unwrapped).rollout(policy, seed=i)
        print(f'episode {i}: reward={rew:.1f}, steps={steps}')
    pyenki.viewer.cleanup()
