from __future__ import annotations

from typing import TYPE_CHECKING

from torchrl.envs.libs.pettingzoo import PettingZooWrapper  # type: ignore

from ...parallel_env import ParallelEnkiEnv
from ...types import PathLike
from .experiment import EnkiExperiment
from .policy import SingleAgentPolicy

if TYPE_CHECKING:
    from torchrl.envs import EnvBase  # type: ignore


def make_env(env: ParallelEnkiEnv,
             seed: int = 0,
             categorical_actions: bool = False) -> EnvBase:
    return PettingZooWrapper(ParallelEnkiEnv(**env.spec),
                             categorical_actions=categorical_actions,
                             device='cpu',
                             seed=seed,
                             return_state=env.has_state)


def reload_experiment(path: PathLike) -> EnkiExperiment:
    return EnkiExperiment.reload_from_file(str(path))


__all__ = [
    'EnkiExperiment',
    'SingleAgentPolicy', 'make_env', 'reload_experiment'
]
