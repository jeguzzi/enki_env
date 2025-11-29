from __future__ import annotations

import pathlib as pl
import sys
from typing import TYPE_CHECKING, cast

import pyenki

from ... import BaseParallelEnv
from ...types import Predictor
from ..utils import load
from .environment import make_env

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv


def make_venv(env: BaseParallelEnv,
              num_envs: int = 1,
              processes: int = 1,
              black_death: bool = False,
              seed: int = 0,
              monitor: bool = True,
              monitor_keywords: tuple[str] = ("is_success", )
              ) -> VecEnv:

    import supersuit
    from stable_baselines3.common.vec_env import VecMonitor

    menv = supersuit.vector.MarkovVectorEnv(env, black_death=black_death)
    venv = supersuit.concat_vec_envs_v1(menv,
                                        num_envs,
                                        num_cpus=processes,
                                        base_class="stable_baselines3")
    for i, env in enumerate(venv.venv.vec_envs):
        env.reset(seed + i)
    if monitor:
        venv = VecMonitor(venv, info_keywords=monitor_keywords)
    return cast('VecEnv', venv)


def train() -> Predictor:
    from stable_baselines3 import SAC

    from ...utils.onnx import export

    venv = make_venv(make_env())
    model = SAC("MultiInputPolicy", venv, verbose=1)
    model.learn(total_timesteps=15_000, log_interval=50, progress_bar=True)
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
    env = make_env(render_mode="human" if display else None)
    for i in range(10):
        data = env.rollout({'': policy}, seed=i)['thymio']
        print(
            f'episode {i}: reward={data.episode_reward:.1f}, steps={data.episode_length}, '
            f'success={data.episode_success[0] if data.episode_success is not None else "?"}'
        )
    if display:
        pyenki.viewer.cleanup()
