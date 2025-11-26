from __future__ import annotations

import pathlib as pl
import sys
from collections.abc import Mapping
from typing import cast

import pyenki

from ...types import Predictor
from ..utils import load
from .environment import make_env


def train() -> Mapping[str, Predictor]:
    import os

    from benchmarl.algorithms import MasacConfig
    from benchmarl.experiment import ExperimentConfig
    from benchmarl.models.mlp import MlpConfig

    from ...utils.benchmarl import EnkiExperiment

    algorithm = MasacConfig.get_from_yaml()
    model = MlpConfig.get_from_yaml()
    config = ExperimentConfig.get_from_yaml()
    config.loggers = ['csv']
    config.render = False
    config.evaluation_interval = 6000
    config.evaluation_episodes = 10
    config.lr = 3e-4
    config.max_n_iters = 10
    config.checkpoint_at_end = True
    config.save_folder = 'logs/MASAC'
    os.makedirs(config.save_folder, exist_ok=True)
    exp = EnkiExperiment(env=make_env(),
                         eval_env=make_env(),
                         config=config,
                         model_config=model,
                         algorithm_config=algorithm,
                         seed=0)
    exp.run_for(30)
    exp.export_policies(pl.Path(__file__).parent, name="masac")
    return exp.get_single_agent_policies()


def get_policies() -> Mapping[str, Predictor]:
    path = pl.Path(__file__).parent
    policies = {
        name: load(path / f"masac_{name}")
        for name in ('thymio', 'e-puck')
    }
    if all(x is not None for x in policies.values()):
        return cast('dict[str, Predictor]', policies)
    return train()


if __name__ == '__main__':
    display = '--display' in sys.argv
    if display:
        import pyenki.viewer
        pyenki.viewer.init()

    policies = get_policies()
    pyenki.viewer.init()
    env = make_env(render_mode="human" if display else None)
    for i in range(10):
        rs = env.rollout(policies, seed=i)
        print(f'episode {i}:')
        for group, data in rs.items():
            print(f'  -{group}: reward={data.episode_reward:.1f}, steps={data.episode_length}')
    if display:
        pyenki.viewer.cleanup()
