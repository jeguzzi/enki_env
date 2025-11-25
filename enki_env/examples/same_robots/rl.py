from __future__ import annotations

import pathlib as pl
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
              monitor: bool = True) -> VecEnv:

    import supersuit
    from stable_baselines3.common.vec_env import VecMonitor

    menv = supersuit.vector.MarkovVectorEnv(env, black_death=black_death)
    venv = supersuit.concat_vec_envs_v1(menv,
                                        num_envs,
                                        num_cpus=processes,
                                        base_class="stable_baselines3")
    for i, env in enumerate(venv.venv.vec_envs):
        env.reset(seed + i)  # type: ignore[misc]
    if monitor:
        venv = VecMonitor(venv)
    return cast('VecEnv', venv)


def train() -> Predictor:
    from stable_baselines3 import SAC

    from ...onnx import export

    venv = make_venv(make_env())
    model = SAC("MultiInputPolicy", venv, verbose=1)
    model.learn(total_timesteps=300_000, log_interval=50, progress_bar=True)
    path = pl.Path(__file__).parent / "sac"
    model.save(path)
    export(model.policy, path.with_suffix(".onnx"))
    return model


def get_policy() -> Predictor:
    return load(pl.Path(__file__).parent / "sac") or train()


if __name__ == '__main__':
    import pyenki.viewer

    policy = get_policy()
    pyenki.viewer.init()
    env = make_env(render_mode="human")
    for i in range(20, 40):
        rew, steps = env.rollout({'': policy}, seed=i)
        print(f'episode {i}: reward={rew:.1f}, steps={steps}')
    pyenki.viewer.cleanup()
