from __future__ import annotations

from typing import Any

import gymnasium as gym

from .parallel_env import ParallelEnkiEnv
from .single_agent_env import SingleAgentEnv


class EnkiEnv(SingleAgentEnv):

    metadata: dict[str, Any] = ParallelEnkiEnv.metadata

    def __init__(self,
                 scenario,
                 config,
                 name: str = '',
                 time_step: float = 0.1,
                 physics_substeps: int = 3,
                 render_mode: str | None = None,
                 render_fps: float = 10.0,
                 render_kwargs: dict[str, Any] = {},
                 notebook: bool | None = None):
        penv = ParallelEnkiEnv(scenario, {name: config}, time_step,
                               physics_substeps, render_mode, render_fps,
                               render_kwargs, notebook)
        super().__init__(penv)


gym.register(
    id="Enki",
    entry_point=EnkiEnv,  # type: ignore
    max_episode_steps=1000,  # Prevent infinite episodes
)
