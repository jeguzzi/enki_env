from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from torchrl.envs.libs.pettingzoo import PettingZooWrapper

from .config import GroupConfig
from .parallel_env import ParallelEnkiEnv
from .types import Scenario


def env(scenario: Scenario,
        config: dict[str, GroupConfig],
        time_step: float = 0.1,
        physics_substeps: int = 3,
        max_duration: float = -1,
        render_mode: str | None = None,
        render_fps: float = 10.0,
        render_kwargs: dict[str, Any] = {},
        notebook: bool | None = None,
        terminate_on: Literal['any', 'all'] | None = 'all',
        device: str = 'cpu',
        seed: int = 0) -> PettingZooWrapper:
    """
    Creates a PyTorchRL environment,
    passing all arguments to :py:class:`enki_env.ParallelEnkiEnv`
    before wrapping it with a :py:func:`torchrl.envs.PettingZooWrapper`.

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
    :param      seed: The random seed passed to :py:func:`torchrl.envs.PettingZooWrapper`.
    :param      device: The device passed to :py:func:`torchrl.envs.PettingZooWrapper`.
    """
    from torchrl.envs.libs.pettingzoo import PettingZooWrapper

    e = ParallelEnkiEnv(scenario,
                        config,
                        time_step=time_step,
                        physics_substeps=physics_substeps,
                        max_duration=max_duration,
                        render_mode=render_mode,
                        render_fps=render_fps,
                        render_kwargs=render_kwargs,
                        notebook=notebook,
                        terminate_on=terminate_on)
    return PettingZooWrapper(
        e,
        categorical_actions=False,
        device=device,
        seed=seed,
        return_state=False,
    )
