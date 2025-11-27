from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

from gymnasium.utils import seeding
from pettingzoo.utils.env import ParallelEnv

from .config import GroupConfig, make_agents, setup_controllers
from .rollout import Rollout
from .types import Action, Array, Info, Observation, Predictor, Scenario
import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym
    from pyenki import DifferentialWheeled, Image, World
    from pyenki.buffer import EnkiRemoteFrameBuffer

StepReturn = tuple[dict[str, Observation], dict[str, float], dict[str, bool],
                   dict[str, bool], dict[str, Info]]
ResetReturn = tuple[dict[str, Observation], dict[str, Info]]

BaseParallelEnv: TypeAlias = ParallelEnv[str, Observation, Action]


def ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]

        return get_ipython() is not None  # type: ignore[no-untyped-call]
    except ImportError:
        return False


class ParallelEnkiEnvSpec(TypedDict):
    """
    Holds the parameters to construct a
    :py:class:`enki_env.ParallelEnkiEnv`.
    """
    scenario: Scenario
    config: dict[str, GroupConfig]
    time_step: float
    physics_substeps: int
    max_duration: float
    render_mode: str | None
    render_fps: float
    render_kwargs: dict[str, Any]
    notebook: bool | None
    terminate_on: Literal['any', 'all'] | None
    success_info: bool
    default_success: bool | None


def parallel_env(
        scenario: Scenario,
        config: dict[str, GroupConfig],
        time_step: float = 0.1,
        physics_substeps: int = 3,
        max_duration: float = -1,
        render_mode: str | None = None,
        render_fps: float = 10.0,
        render_kwargs: dict[str, Any] = {},
        notebook: bool | None = None,
        terminate_on: Literal['any', 'all'] | None = 'all',
        success_info: bool = True,
        default_success: bool | None = None) -> BaseParallelEnv:
    """
    Helper function that creates a parallel environment,
    passing all arguments to :py:class:`enki_env.ParallelEnkiEnv`.

    :param      scenario:          The scenario that generates worlds
        at :py:meth:`gymnasium.Env.reset`
    :param      config:            The configuration for all groups. Robots
        with a :py:attr:`pyenki.PhysicalObject.name` corresponding to the group,
        will be configured accordingly. The empty group name `""`
        will catch the remaining robots, if defined.
    :param      name:              The name of the robot
    :param      time_step:         The time step of the simulation [s]
    :param      max_duration:      The maximum duration of the episodes [s]
    :param      physics_substeps:  The number of physics sub-steps for each
        simulation step, see :py:meth:`pyenki.World.step`.
    :param      render_mode:       The render mode (one of ``None``,
        ``rgb_array`` or ``human``).
    :param      render_fps:        The render fps (only relevant
        when ``render_mode="human"``.
    :param      render_kwargs:     The render keywords arguments
        arguments forwarded to :py:func:`pyenki.viewer.render` when
        rendering an environment.
    :param      notebook:          Whether the we should use a notebook-compatible
        renderer. If ``None``, it will check if we are running a notebook.
    :param      terminate_on:      Whether to terminate the episode as soon as the first agent
        terminates ("any") or whether to wait for all agents to terminate before
        removing all of them at once. If set to ``None``, it will terminate each
        agent individually, removing them from the environment independently from each other.
    :param      success_info:      Whether include key ``"is_success"`` in the final info dictionary for each robot.
        It will be included only if set by one of :py:attr:`enki_env.GroupConfig.terminations`
        or if ``default_success`` is not ``None``.
    :param      default_success:   The value associated to ``"is_success"`` in the final info dictionary
        when the robot has not been terminated.
    """
    env: BaseParallelEnv = ParallelEnkiEnv(scenario,
                                           config,
                                           time_step=time_step,
                                           physics_substeps=physics_substeps,
                                           max_duration=max_duration,
                                           render_mode=render_mode,
                                           render_fps=render_fps,
                                           render_kwargs=render_kwargs,
                                           notebook=notebook,
                                           terminate_on=terminate_on,
                                           success_info=success_info,
                                           default_success=default_success)
    return env


class ParallelEnkiEnv(BaseParallelEnv):
    """
    A :py:class:`pettingzoo.utils.env.ParallelEnv` that exposes robots in
    a :py:class:`pyenki.World`.

    Observations, rewards, and information returned by :py:meth:`gymnasium.Env.reset` and
    :py:meth:`gymnasium.Env.step`,
    are generated from the robot sensors and internal state using
    :py:attr:`enki_env.GroupConfig.observation`, :py:attr:`enki_env.GroupConfig.reward`,
    and :py:attr:`enki_env.GroupConfig.info`.
    Termination criteria are specified :py:attr:`enki_env.GroupConfig.terminations`.
    Actions are actuated according to :py:attr:`enki_env.GroupConfig.action`.

    Rendering is performed:

    - by a :py:class:`pyenki.viewer.WorldView` if ``render_mode="human"``
      and we are not running in a Jupyter notebook.
    - by a :py:class:`pyenki.buffer.EnkiRemoteFrameBuffer` if ``render_mode="human"``
      and we are running in a Jupyter notebook.
    - by :py:func:`pyenki.viewer.render` if ``render_mode="rgb_array"``.

    To create an environment, you need to

    1. define a scenario with a least one robot, e.g. ::

            import pyenki

            def my_scenario(seed: int) -> pyenki.World:
                world = pyenki.World(seed)
                world.add_object(pyenki.Thymio2())
                world.add_object(pyenki.EPuck())
                return world

    2. define a configuration, e.g., the default configuration associated to the robot ::

            import enki_env

            configs = {'thymio': enki_env.ThymioConfig(),
                       'e-puck': enki_env.EPuckConfig()}

    3. call the factory function, customizing the other parameters as you see fit ::

            env = enki_env.parallel_env(scenario, configs, max_duration=10)
    """

    metadata: dict[str, Any] = {"render_modes": ['human', 'rgb_array']}
    render_mode: str | None = None

    # Created
    _np_random: np.random.Generator | None = None
    # will be set to the "invalid" value -1 if the seed of the currently set rng is unknown
    _np_random_seed: int | None = None

    @property
    def config(self) -> dict[str, GroupConfig]:
        return self._config

    def display_in_notebook(self) -> None:
        """
        Display the environment in a notebook using a
        an interactive :py:class:`pyenki.buffer.EnkiRemoteFrameBuffer`.

        Requires ``render_mode="human"`` and a notebook.
        """
        from IPython.display import display

        if self._render_buffer:
            display(self._render_buffer)  # type: ignore[no-untyped-call]
        else:
            if self.render_mode != "human":
                warnings.warn('render_mode not set to "human"', stacklevel=2)
            else:
                warnings.warn('Requires running in a notebook', stacklevel=2)

    def snapshot(self) -> None:
        """
        Display the environment in a notebook.

        Requires ``render_mode="human"`` and a notebook.
        """
        from IPython.display import display

        if self._render_buffer:
            display(self._render_buffer.snapshot())  # type: ignore[no-untyped-call]
        else:
            if self.render_mode != "human":
                warnings.warn('render_mode not set to "human"', stacklevel=2)
            else:
                warnings.warn('Requires running in a notebook', stacklevel=2)

    def observation_space(self, agent: str) -> gym.spaces.Space[Any]:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Space[Any]:
        return self.action_spaces[agent]

    @property
    def possible_agents(self) -> list[str]:
        return self._possible_agents

    @possible_agents.setter
    def possible_agents(self, value: Sequence[str]) -> None:
        raise NotImplementedError()

    @property
    def agents(self) -> list[str]:
        return list(self._agents)

    @agents.setter
    def agents(self, value: Sequence[str]) -> None:
        raise NotImplementedError()

    def _get_observations(self) -> dict[str, Observation]:
        return {
            uid: config.observation.get(agent)
            for uid, (agent, _, config) in self._agents.items()
        }

    def _get_rewards(self) -> dict[str, float]:
        if self._world:
            return {
                uid:
                config.reward(agent, self._world) if config.reward else -1
                for uid, (agent, _, config) in self._agents.items()
            }
        else:
            return {}

    def _get_infos(self) -> dict[str, Info]:
        if self._world:
            return {
                uid:
                config.info(agent, self._world) if config.info else {}
                for uid, (agent, _, config) in self._agents.items()
            }
        else:
            return {}

    def _update_truncations(self) -> dict[str, bool]:
        value = self._max_duration > 0 and self._duration >= self._max_duration
        return {uid: value for uid in self._agents}

    def _update_terminations(self) -> dict[str, bool]:
        ts: dict[str, bool] = {}
        if not self._world:
            return ts
        for uid, (agent, _, config) in self._agents.items():
            if uid in self._success:
                ts[uid] = True
            else:
                ts[uid] = False
                for t in config.terminations:
                    r = t(agent, self._world)
                    if r is not None:
                        ts[uid] = True
                        self._success[uid] = r
                        break
        if self._terminate_on == 'any' and any(ts.values()):
            return {uid: True for uid in self._agents}
        if self._terminate_on == 'all' and not all(ts.values()):
            return {uid: False for uid in self._agents}
        return ts

    def _actuate(self, acts: dict[str, Action], dt: float) -> None:
        for uid, act in acts.items():
            if uid not in self.agents:
                continue
            agent, _, config = self._agents[uid]
            config.action.actuate(act, agent, dt)

    @property
    def has_state(self) -> bool:
        return False

    @property
    def spec(self) -> ParallelEnkiEnvSpec:
        """
        The parameters used to construct the environment.
        """
        return self._spec

    def __init__(self,
                 scenario: Scenario,
                 config: dict[str, GroupConfig],
                 time_step: float = 0.1,
                 physics_substeps: int = 3,
                 max_duration: float = -1,
                 render_mode: str | None = None,
                 render_fps: float = 10.0,
                 render_kwargs: dict[str, Any] = {},
                 notebook: bool | None = None,
                 terminate_on: Literal['any', 'all'] | None = 'all',
                 success_info: bool = True,
                 default_success: bool | None = None):
        """
        Constructs a new instance. Similar arguments
        as :py:class:`enki_env.ParallelEnkiEnv` referring to a single robot.

        :param      scenario:          The scenario that generates worlds
            at :py:meth:`gymnasium.Env.reset`
        :param      config:            The configuration for all groups. Robots
            with a :py:attr:`pyenki.PhysicalObject.name` corresponding to the group,
            will be configured accordingly. The empty group name `""`
            will catch the remaining robots, if defined.
        :param      name:              The name of the robot
        :param      time_step:         The time step of the simulation [s]
        :param      max_duration:      The maximum duration of the episodes [s]
        :param      physics_substeps:  The number of physics sub-steps for each
            simulation step, see :py:meth:`pyenki.World.step`.
        :param      render_mode:       The render mode (one of ``None``,
            ``rgb_array`` or ``human``).
        :param      render_fps:        The render fps (only relevant
            when ``render_mode="human"``.
        :param      render_kwargs:     The render keywords arguments
            arguments forwarded to :py:func:`pyenki.viewer.render` when
            rendering an environment.
        :param      notebook:          Whether the we should use a notebook-compatible
            renderer. If ``None``, it will check if we are running a notebook.
        :param      terminate_on:      Whether to terminate the episode as soon as the first agent
            terminates ("any") or whether to wait for all agents to terminate before
            removing all of them at once. If set to ``None``, it will terminate each
            agent individually, removing them from the environment independently from each other.
        :param      success_info:      Whether include key ``"is_success"`` in the final info dictionary for each robot.
            It will be included only if set by one of :py:attr:`enki_env.GroupConfig.terminations`
            or if ``default_success`` is not ``None``.
        :param      default_success:   The value associated to ``"is_success"`` in the final info dictionary
            when the robot has not been terminated.
        """
        if notebook is None:
            notebook = ipython()
        self._spec: ParallelEnkiEnvSpec = dict(
            scenario=scenario,
            config=config,
            time_step=time_step,
            physics_substeps=physics_substeps,
            max_duration=max_duration,
            render_mode=render_mode,
            render_fps=render_fps,
            render_kwargs=render_kwargs,
            notebook=notebook,
            terminate_on=terminate_on,
            success_info=success_info,
            default_success=default_success)
        self._scenario = scenario
        self._config = config
        self._time_step = time_step
        self._physics_substeps = physics_substeps
        self.render_mode = render_mode
        self._render_kwargs = render_kwargs
        self._render_fps = render_fps
        self._max_duration = max_duration
        self._terminate_on = terminate_on
        self._duration: float = 0
        self._success_info = success_info
        self._default_success = default_success
        world = scenario(0)
        agents = make_agents(world, config)
        self._possible_agents = list(agents)
        self._agents: dict[str, tuple[DifferentialWheeled, str,
                                      GroupConfig]] = {}
        self.group_observation_spaces = {
            group: config.observation.space
            for group, config in config.items()
        }
        self.group_action_spaces = {
            group: config.action.space
            for group, config in config.items()
        }
        self.observation_spaces = {
            uid: config.observation.space
            for uid, (_, _, config) in agents.items()
        }
        self.action_spaces = {
            uid: config.action.space
            for uid, (_, _, config) in agents.items()
        }
        self._world: World | None = None
        self._world_view = None
        self._render_buffer: EnkiRemoteFrameBuffer | None = None
        if self.render_mode == 'human':
            if notebook:
                from pyenki.buffer import EnkiRemoteFrameBuffer

                # print('Create render buffer')
                self._render_buffer = EnkiRemoteFrameBuffer(
                    world=None, **self._render_kwargs)
            else:
                from pyenki.viewer import WorldView

                # print('Create a Qt world view')
                self._world_view = WorldView(world=None, **self._render_kwargs)
                self._world_view.show()

    def render(self) -> Image | None:
        from pyenki.viewer import render

        if self._world:
            return render(self._world, **self._render_kwargs)
        return None

    @property
    def np_random_seed(self) -> int:
        """
        Returns the environment's internal :attr:`_np_random_seed`
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
        """
        Returns the environment's internal :attr:`_np_random`
        that if not set will initialise with a random seed.

        Returns:
            Instances of `np.random.Generator`
        """
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator) -> None:
        self._np_random = value
        self._np_random_seed = -1

    @property
    def group_map(self) -> dict[str, list[str]]:
        """
        The names of the robots belonging to each group.
        """
        groups: dict[str, list[str]] = {}
        for name, (_, group, _) in self._agents.items():
            if group not in groups:
                groups[group] = [name]
            else:
                groups[group].append(name)
        return groups

    @property
    def _agent_groups(self) -> dict[str, str]:
        return {agent: group for agent, (_, group, _) in self._agents.items()}

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> ResetReturn:
        world = self._world
        # Same as gymnasium.Env.reset
        # Initialize the RNG if the seed is manually passed

        if seed is not None and seed >= 0:
            self._np_random, self._np_random_seed = seeding.np_random(seed)
            world_seed = seed
        elif world:
            world_seed = world.random_seed
        else:
            world_seed = self.np_random_seed
        world_seed &= (2**63 - 1)
        self._world = self._scenario(world_seed)
        assert self._world
        if seed is None and world:
            self._world.copy_random_generator(world)
        if self._world_view:
            self._world_view.world = self._world
        if self._render_buffer:
            self._render_buffer.world = self._world
        self._world.step(self._time_step, 1)
        self._agents = make_agents(self._world, self._config)
        self._success: dict[str, bool] = {}
        self._duration = 0
        return self._get_observations(), self._get_infos()

    def step(self, actions: dict[str, Action]) -> StepReturn:
        assert self._world
        self._actuate(actions, self._time_step)
        self._world.step(self._time_step, self._physics_substeps)
        obs = self._get_observations()
        infos = self._get_infos()
        rew = self._get_rewards()
        self._duration += self._time_step
        trunc = self._update_truncations()
        term = self._update_terminations()

        for uid in list(self._agents):
            if trunc[uid] or term[uid]:
                if self._success_info:
                    success = self._success.get(uid, self._default_success)
                    if success is not None:
                        infos[uid]['is_success'] = np.asarray(success, dtype=np.bool_)
                del self._agents[uid]

        if self.render_mode == "human":
            if self._render_buffer:
                self._render_buffer.tick(self._render_fps)
            if self._world_view:
                from pyenki.viewer import run

                run(1 / self._render_fps)
        return obs, rew, term, trunc, infos

    def state(self) -> Array:
        raise NotImplementedError()

    def close(self) -> None:
        if self._world_view:
            self._world_view.hide()
            del self._world_view

    def make_world(self,
                   policies: dict[str, Predictor],
                   seed: int = 0) -> World:
        """
        Generate a world using the scenario and optionally
        assign a policy to the robots controller.

        :param      policies:  The policies assigned to groups
        :param      seed:      The random seed

        :returns:   The world
        """
        world = self._scenario(seed)
        setup_controllers(world, self._config, policies)
        return world

    # TODO(Jerome): seed action spaces
    def rollout(self,
                policies: Mapping[str, Predictor] = {},
                max_steps: int = -1,
                seed: int = 0) -> dict[str, Rollout]:
        """
        Performs a rollout of an episode

        :param      policies:   The policies assigned to groups.
            If a group misses a policy, it will randomly generate actions.
        :param      max_steps:  The maximum steps to perform
        :param      seed:       The random seed

        :returns:   A dictionary keyed by groups with
            the data collected during the rollout
        """
        actions: list[dict[str, Action]] = []
        observations: list[dict[str, Observation]] = []
        rewards: list[dict[str, float]] = []
        infos: list[dict[str, Info]] = []
        terminations: list[dict[str, bool]] = []
        truncations: list[dict[str, bool]] = []
        obs, _ = self.reset(seed=seed)
        observations.append(obs)
        step = 0
        agent_policies: dict[str, Predictor | None] = {
            agent: policies.get(group) or policies.get('')
            for agent, group in self._agent_groups.items()
        }
        group_map = self.group_map
        while (max_steps <= 0 or step < max_steps) and self.agents:
            step += 1
            act: dict[str, Action] = {}
            for (agent, o) in obs.items():
                policy = agent_policies[agent]
                if policy:
                    act[agent] = policy.predict(o, deterministic=True)[0]
                else:
                    act[agent] = self.action_spaces[agent].sample()
            actions.append(act)
            obs, rew, term, trunc, info = self.step(act)
            observations.append(obs)
            rewards.append(rew)
            terminations.append(term)
            truncations.append(trunc)
            infos.append(info)
        return {
            group:
            Rollout.aggregate(agents, self.group_action_spaces[group],
                              self.group_observation_spaces[group], actions,
                              observations, rewards, terminations, truncations,
                              infos)
            for group, agents in group_map.items()
        }
