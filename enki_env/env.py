from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, cast

import gymnasium as gym

from .config import GroupConfig
from .parallel_env import ParallelEnkiEnv
from .rollout import Rollout
from .single_agent_env import SingleAgentEnv
from .types import Action, Observation, Predictor, Scenario

if TYPE_CHECKING:
    from pyenki import World

BaseEnv: TypeAlias = gym.Env[Observation, Action]


class EnkiEnv(SingleAgentEnv[str, Observation, Action]):

    metadata: dict[str, Any] = ParallelEnkiEnv.metadata

    def __init__(self,
                 scenario: Scenario,
                 config: GroupConfig,
                 name: str = '',
                 time_step: float = 0.1,
                 max_duration: float = -1,
                 physics_substeps: int = 3,
                 render_mode: str | None = None,
                 render_fps: float = 10.0,
                 render_kwargs: dict[str, Any] = {},
                 notebook: bool | None = None) -> None:
        penv = ParallelEnkiEnv(scenario, {name: config}, time_step,
                               physics_substeps, max_duration, render_mode,
                               render_fps, render_kwargs, notebook)
        super().__init__(penv)

    def make_world(self, policy: Predictor, seed: int = 0) -> World:
        return cast('ParallelEnkiEnv', self._penv).make_world({'': policy},
                                                              seed=seed)

    def rollout(self,
                policy: Predictor | None,
                max_steps: int = -1,
                seed: int = 0) -> Rollout:
        rs = cast('ParallelEnkiEnv',
                  self._penv).rollout(policies={'': policy} if policy else {},
                                      max_steps=max_steps,
                                      seed=seed)
        assert len(rs) == 1
        return next(iter(rs.values()))


gym.register(
    id="Enki",
    entry_point=EnkiEnv,  # type: ignore
    max_episode_steps=1000,  # Prevent infinite episodes
)
