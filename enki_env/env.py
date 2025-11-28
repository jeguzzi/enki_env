from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias, cast

import gymnasium as gym

from .config import GroupConfig
from .parallel_env import ParallelEnkiEnv
from .rollout import Rollout
from .scenario import Scenario
from .single_agent_env import SingleAgentEnv
from .types import Action, Observation, Predictor

if TYPE_CHECKING:
    import pyenki

BaseEnv: TypeAlias = gym.Env[Observation, Action]


class EnkiEnv(SingleAgentEnv[str, Observation, Action]):
    """
    A :py:class:`gymnasium.Env` that exposes a single robot in
    a :py:class:`pyenki.World`.

    Internally, it creates a :py:class:`enki_env.ParallelEnkiEnv`
    with a single group composed by the robot, and then forwards to it methods like
    :py:meth:`gymnasium.Env.reset`, :py:meth:`gymnasium.Env.step`,
    and :py:meth:`gymnasium.Env.render`.

    Observations, rewards, and information returned by :py:meth:`gymnasium.Env.reset` and
    :py:meth:`gymnasium.Env.step`, are generated from the robot sensors and internal state using
    :py:attr:`enki_env.GroupConfig.observation`, :py:attr:`enki_env.GroupConfig.reward`,
    and :py:attr:`enki_env.GroupConfig.info`.
    Termination criteria are specified in :py:attr:`enki_env.GroupConfig.terminations`.
    Actions are actuated according to :py:attr:`enki_env.GroupConfig.action`.

    Rendering is performed:

    - by a :py:class:`pyenki.viewer.WorldView` if ``render_mode="human"``
      and we are not in a Jupyter notebook.
    - by a :py:class:`pyenki.buffer.EnkiRemoteFrameBuffer` if ``render_mode="human"``
      and we are in a Jupyter notebook.
    - by :py:func:`pyenki.viewer.render` if ``render_mode="rgb_array"``.

    The environment is registered under id ``"Enki"``. To create an environment,
    you need to first

    1. define a scenario with a least one robot, e.g. ::

            import enki_env
            import pyenki

            class MyScenario(enki_env.BaseScenario):

                def init(self, world: pyenki.World) -> None:
                    robot = pyenki.Thymio2()
                    robot.angle = world.random_generator.uniform(-1, 1)
                    world.add_object(robot)

    2. define a configuration, e.g., the default configuration associated with the robot ::

            config = enki_env.ThymioConfig()

    Then, you can call the factory function, customizing the other parameters as you see fit ::

        import gymnasium

        env = gymnasium.make("Enki", MyScenario(), config, max_duration=10)

    """

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
                 notebook: bool | None = None,
                 success_info: bool = True,
                 default_success: bool | None = None) -> None:
        """
        Constructs a new instance. It takes the same arguments
        as :py:class:`enki_env.ParallelEnkiEnv` but referring to a single robot.

        :param      scenario:          The scenario that generates worlds
            at :py:meth:`gymnasium.Env.reset`.
        :param      config:            The configuration for the group containing the robot.
        :param      name:              The name of the robot.
        :param      time_step:         The time step of the simulation [s].
        :param      max_duration:      The maximum duration of the episodes [s].
        :param      physics_substeps:  The number of physics sub-steps for each
            simulation step, see :py:meth:`pyenki.World.step`.
        :param      render_mode:       The render mode (one of ``None``,
            ``rgb_array`` or ``human``).
        :param      render_fps:        The render fps (only relevant
            when ``render_mode="human"``.
        :param      render_kwargs:     The render keywords arguments
            arguments forwarded to :py:func:`pyenki.viewer.render` when
            rendering an environment.
        :param      notebook:          Whether to use a notebook-compatible
            renderer. If ``None``, it will select it if we are running a notebook.
        :param      success_info:      Whether to include key ``"is_success"``
            in the final info dictionary for each robot. It will be included only if
            it has been set by one of :py:attr:`enki_env.GroupConfig.terminations`
            or if ``default_success`` is not ``None``.
        :param      default_success:   The value associated with ``"is_success"`` in the
            final info dictionary when, at the end of the episode,
            the robot has not been yet terminated.
        """
        penv = ParallelEnkiEnv(scenario=scenario,
                               config={name: config},
                               time_step=time_step,
                               physics_substeps=physics_substeps,
                               max_duration=max_duration,
                               render_mode=render_mode,
                               render_fps=render_fps,
                               render_kwargs=render_kwargs,
                               notebook=notebook,
                               success_info=success_info,
                               default_success=default_success)
        super().__init__(penv)

    @property
    def config(self) -> GroupConfig:
        """
        The robot configuration.
        """
        rs = cast('ParallelEnkiEnv', self._penv).config
        assert len(rs) == 1
        return next(iter(rs.values()))

    def display_in_notebook(self) -> None:
        """
        Displays the environment in a notebook using a
        an interactive :py:class:`pyenki.buffer.EnkiRemoteFrameBuffer`.

        Requires ``render_mode="human"`` and a notebook.
        """
        return cast('ParallelEnkiEnv', self._penv).display_in_notebook()

    def snapshot(self) -> None:
        """
        Displays the environment in a notebook.

        Requires ``render_mode="human"`` and a notebook.
        """
        return cast('ParallelEnkiEnv', self._penv).snapshot()

    def make_world(self,
                   policy: Predictor | None = None,
                   seed: int = 0,
                   deterministic: bool = True) -> pyenki.World:
        """
        Generates a world using the scenario and, if specified,
        assign a policy to the robot controller.

        :param      policy:  The policy
        :param      seed:    The random seed
        :param      deterministic: Whether to evaluate the policy
            deterministically.

        :returns:   The world
        """
        return cast('ParallelEnkiEnv',
                    self._penv).make_world({'': policy} if policy else {},
                                           seed=seed,
                                           deterministic=deterministic)

    def rollout(self,
                policy: Predictor | None = None,
                max_steps: int = -1,
                seed: int = 0,
                deterministic: bool = True) -> Rollout:
        """
        Performs the rollout of an episode

        :param      policy:     The policy to apply; if not provided,
            it will randomly generate actions.
        :param      max_steps:  The maximum number of steps to perform.
        :param      seed:       The random seed.
        :param      deterministic: Whether to evaluate the policy
            deterministically.

        :returns:   The data collected during the rollout.
        """
        rs = cast('ParallelEnkiEnv',
                  self._penv).rollout(policies={'': policy} if policy else {},
                                      max_steps=max_steps,
                                      seed=seed,
                                      deterministic=deterministic)
        assert len(rs) == 1
        return next(iter(rs.values()))


gym.register(
    id="Enki",
    entry_point=EnkiEnv,  # type: ignore
    max_episode_steps=1000,  # Prevent infinite episodes
)
