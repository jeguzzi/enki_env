from __future__ import annotations

from .config import GroupConfig, setup_controllers
from .env import EnkiEnv, BaseEnv
from .parallel_env import ParallelEnkiEnv, BaseParallelEnv, parallel_env
from .robots import (EPuckAction, EPuckConfig, EPuckObservation, MarxbotAction,
                     MarxbotConfig, MarxbotObservation, ThymioAction,
                     ThymioConfig, ThymioObservation)


def default_config() -> dict[str, GroupConfig]:
    """
    Return a dictionary where default robots' names are
    associated to robots' default configurations, i.e.:

    - thymio -> :py:class:`enki_env.ThymioConfig`
    - e-puck -> :py:class:`enki_env.EPuckConfig`
    - marxbot -> :py:class:`enki_env.MarxbotConfig`

    :returns: the dictionary of default configurations
    """
    rs: dict[str, GroupConfig] = {
        'thymio': ThymioConfig(),
        'e-puck': EPuckConfig(),
        'marxbot': MarxbotConfig(),
    }
    return rs


__all__ = [
    'GroupConfig', 'setup_controllers', 'EnkiEnv', 'ParallelEnkiEnv',
    'EPuckAction', 'EPuckObservation', 'MarxbotAction', 'MarxbotObservation',
    'ThymioAction', 'ThymioObservation', 'EPuckConfig', 'MarxbotConfig',
    'ThymioConfig', 'BaseEnv', 'BaseParallelEnv', 'parallel_env'
]
