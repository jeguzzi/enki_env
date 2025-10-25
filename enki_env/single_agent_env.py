from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from gymnasium import Env
from pettingzoo.utils.env import ParallelEnv  # type: ignore[import-untyped]

from .types import Info

IndexType = TypeVar('IndexType')
ObservationType = TypeVar('ObservationType')
ActionType = TypeVar('ActionType')


class SingleAgentEnv(Env[ObservationType, ActionType]):

    # spec: EnvSpec | None = None

    def __init__(self, penv: ParallelEnv[IndexType, ObservationType,
                                         ActionType]):
        self._penv = penv
        self.action_space = next(iter(self._penv.action_spaces.values()))
        self.observation_space = next(
            iter(self._penv.observation_spaces.values()))
        self.metadata = self._penv.metadata
        self.render_mode = self._penv.render_mode

    @property
    def parallel_env(
            self) -> ParallelEnv[IndexType, ObservationType, ActionType]:
        return self._penv

    @property
    def np_random_seed(self) -> int:
        return self._penv.np_random_seed

    @property
    def np_random(self) -> np.random.Generator:
        return self._penv.np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._penv.np_random = value

    # @property
    # def metadata(self) -> dict[str, Any]:
    #     return self._penv.metadata

    # @property
    # def render_mode(self) -> str | None:
    #     return self._penv.render_mode

    # @property
    # def action_space(self) -> spaces.Space:
    #     return next(iter(self._penv.action_spaces.values()))

    # @property
    # def observation_space(self) -> spaces.Space:
    #     return next(iter(self._penv.observation_spaces.values()))

    def step(
            self, action: ActionType
    ) -> tuple[ObservationType, float, bool, bool, Info]:
        agent = self._penv.agents[0]
        obss, rews, terms, truncs, infos = self._penv.step({agent: action})
        agent = self._penv.agents[0]
        return obss[agent], rews[agent], terms[agent], truncs[agent], infos[
            agent]

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None
    ) -> tuple[ObservationType, Info]:
        super().reset(seed=seed)
        obss, infos = self._penv.reset(seed, options)
        agent = self._penv.agents[0]
        return obss[agent], infos[agent]

    def render(self):
        self._penv.render()

    def close(self) -> None:
        self._penv.close()

    def has_wrapper_attr(self, name: str) -> bool:
        return hasattr(self._penv, name)

    def get_wrapper_attr(self, name: str) -> Any:
        return getattr(self._penv, name)

    def set_wrapper_attr(self,
                         name: str,
                         value: Any,
                         *,
                         force: bool = True) -> bool:
        if force or hasattr(self, name):
            setattr(self._penv, name, value)
            return True
        return False
