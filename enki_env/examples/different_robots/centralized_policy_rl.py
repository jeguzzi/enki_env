from __future__ import annotations

import pathlib as pl
import sys

import pyenki

from ...types import Predictor
from ..utils import load
from .environment import make_env
from ...concat_env import ConcatEnv


def train() -> Predictor:
    from stable_baselines3 import SAC

    from ...utils.onnx import export

    env = ConcatEnv(make_env())
    model = SAC("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=15_000, log_interval=50, progress_bar=True)
    path = pl.Path(__file__).parent / "centralized_sac"
    model.save(path)
    export(model.policy, path.with_suffix(".onnx"))
    return model


def get_policy() -> Predictor:
    return load(pl.Path(__file__).parent / "centralized_sac") or train()


if __name__ == '__main__':
    display = '--display' in sys.argv
    if display:
        import pyenki.viewer
        pyenki.viewer.init()

    policy = get_policy()
    env = ConcatEnv(make_env(render_mode="human" if display else None))
    for i in range(10):
        rs = env.rollout(policy=policy, seed=i)
        print(f'episode {i}:')
        for group, data in rs.items():
            print(
                f'  -{group}: reward={data.episode_reward:.1f}, '
                f'steps={data.episode_length}, '
                f'success={data.episode_success[0] if data.episode_success is not None else "?"}'
            )
    if display:
        pyenki.viewer.cleanup()
