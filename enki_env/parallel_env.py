from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from pettingzoo.utils.env import ParallelEnv  # type: ignore[import-untyped]
from pyenki import DifferentialWheeled, World, WorldView, run_ui

from .config import GroupConfig, make_agents
from .types import Action, Array, Info, Observation, Scenario

StepReturn = tuple[dict[str, Observation], dict[str, float], dict[str, bool],
                   dict[str, bool], dict[str, Info]]
ResetReturn = tuple[dict[str, Observation], dict[str, Info]]


def ipython():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


class ParallelEnkiEnv(ParallelEnv[str, Observation, Action]):

    metadata: dict[str, Any] = {"render_modes": ['human', 'rgb_array']}
    render_mode: str | None = None

    # Created
    _np_random: np.random.Generator | None = None
    # will be set to the "invalid" value -1 if the seed of the currently set rng is unknown
    _np_random_seed: int | None = None

    @property
    def possible_agents(self) -> list[str]:
        return self._possible_agents

    @property
    def agents(self) -> list[str]:
        return list(self._agents)

    @property
    def action_spaces(self) -> dict[str, gym.spaces.Box]:
        return self._action_space

    @property
    def observation_spaces(self) -> dict[str, gym.spaces.Dict]:
        return self._observation_space

    def action_space(self, agent: str) -> gym.spaces.Box:
        return self._action_space[agent]

    def observation_space(self, agent: str) -> gym.spaces.Dict:
        return self._observation_space[agent]

    def _get_observations(self) -> dict[str, Observation]:
        return {
            uid: config.observation.get(agent)
            for uid, (agent, _, config) in self._agents.items()
        }

    def _get_rewards(self) -> dict[str, float]:
        if self._world:
            return {
                uid: config.reward.get(agent, self._world)
                for uid, (agent, _, config) in self._agents.items()
            }
        else:
            return {}

    def _get_infos(self) -> dict[str, Info]:
        if self._world:
            return {
                uid: config.info.get(agent, self._world)
                for uid, (agent, _, config) in self._agents.items()
            }
        else:
            return {}

    def _update_terminations(self) -> dict[str, bool]:
        ts: dict[str, bool] = {}
        if not self._world:
            return ts
        for uid, (agent, _, config) in list(self._agents.items()):
            ts[uid] = False
            for t in config.terminations:
                r = t(agent, self._world)
                if r is not None:
                    ts[uid] = True
                    self._success[uid] = r
                    del self._agents[uid]
                    break
        return ts

    def _actuate(self, acts: dict[str, Action], dt: float) -> None:
        for uid, act in acts.items():
            if uid not in self.agents:
                continue
            agent, _, config = self._agents[uid]
            config.action.actuate(act, agent, dt)

    def __init__(self,
                 scenario: Scenario,
                 config: dict[str, GroupConfig],
                 time_step: float = 0.1,
                 physics_substeps: int = 3,
                 render_mode: str | None = None,
                 render_fps: float = 10.0,
                 render_kwargs: dict[str, Any] = {},
                 notebook: bool | None = None):
        if notebook is None:
            notebook = ipython()
        self._scenario = scenario
        self._config = config
        self._time_step = time_step
        self._physics_substeps = physics_substeps
        self.render_mode = render_mode
        self._render_kwargs = render_kwargs
        self._render_fps = render_fps
        world = scenario(None)
        agents = make_agents(world, config)
        self._possible_agents = list(agents)
        self._agents: dict[str, tuple[DifferentialWheeled, str,
                                      GroupConfig]] = {}
        self._observation_space = {
            uid: config.observation.space
            for uid, (_, _, config) in agents.items()
        }
        self._action_space = {
            uid: config.action.space
            for uid, (_, _, config) in agents.items()
        }
        self._world: World | None = None
        self._render_buffer = None
        self._world_view = None
        if self.render_mode == 'human':
            if notebook:
                from pyenki.buffer import EnkiRemoteFrameBuffer

                print('Create render buffer')
                self._render_buffer = EnkiRemoteFrameBuffer(
                    world=None, **self._render_kwargs)
            else:
                print('Create qt world view')
                self._world_view = WorldView(world=None, **self._render_kwargs)

    def render(self) -> Array | None:
        if self._world:
            return self._world.render(**self._render_kwargs)
        return None

    @property
    def np_random_seed(self) -> int:
        """Returns the environment's internal :attr:`_np_random_seed`
           that if not set will first initialise with a random int as seed.

        If :attr:`np_random_seed` was set directly instead of
        through :meth:`reset` or :meth:`set_np_random_through_seed`,
        the seed will take the value -1.

        Returns:
            int: the seed of the current `np_random` or -1,
            if the seed of the rng is unknown
        """
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random_seed

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random`
        that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value
        self._np_random_seed = -1

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> ResetReturn:
        # Same as gymnasium.Env.reset
        # Initialize the RNG if the seed is manually passed
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
        world = self._world
        self._world = self._scenario(self._np_random)
        assert self._world
        if seed is not None:
            # TODO
            pass
            # self._world.set_random_seed(seed)
        elif world:
            # TODO
            pass
            # self._world.copy_random_generator(world)
        else:
            pass
            # self._world.set_random_seed(self.np_random_seed & (2**63 - 1))
        if self._world_view:
            self._world_view.world = self._world
        if self._render_buffer:
            self._render_buffer.world = self._world
        self._world.step(self._time_step, 1)
        self._agents = make_agents(self._world, self._config)
        self._success: dict[str, bool] = {}
        return self._get_observations(), self._get_infos()

    def step(self, actions: dict[str, Action]) -> StepReturn:
        assert self._world
        self._actuate(actions, self._time_step)
        self._world.step(self._time_step, self._physics_substeps)
        obs = self._get_observations()
        infos = self._get_infos()
        rew = self._get_rewards()
        trunc = {uid: False for uid in self._agents}
        term = self._update_terminations()
        if self.render_mode == "human":
            if self._render_buffer:
                self._render_buffer.tick(self._render_fps)
            if self._world_view:
                run_ui(1 / self._render_fps)
        return obs, rew, term, trunc, infos

    def state(self) -> Array:
        raise NotImplementedError()

    def close(self) -> None:
        if self._world_view:
            self._world_view.hide()
            del self._world_view
