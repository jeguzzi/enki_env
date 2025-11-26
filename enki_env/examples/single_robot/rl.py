from __future__ import annotations

import pathlib as pl
import sys
from typing import cast

import pyenki

from ... import EnkiEnv
from ...types import Predictor
from ..utils import load
from .environment import make_env


def train() -> Predictor:
    from stable_baselines3 import SAC

    from ...onnx import export

    env = make_env()
    model = SAC("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000, log_interval=50, progress_bar=True)
    path = pl.Path(__file__).parent / "sac"
    model.save(path)
    export(model.policy, path.with_suffix(".onnx"))
    return model


def get_policy() -> Predictor:
    return load(pl.Path(__file__).parent / "sac") or train()


if __name__ == '__main__':
    display = '--display' in sys.argv
    if display:
        import pyenki.viewer
        pyenki.viewer.init()

    policy = get_policy()
    pyenki.viewer.init()
    env = make_env(render_mode="human" if display else None)
    for i in range(10):
        data = cast('EnkiEnv', env.unwrapped).rollout(policy, seed=i)
        print(f'episode {i}: reward={data.episode_reward:.1f}, steps={data.episode_length}')
    if display:
        pyenki.viewer.cleanup()
