from __future__ import annotations

import dataclasses as dc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, SupportsFloat, cast

import numpy as np

from .info import InfoFunction
from .reward import RewardFunction
from .types import Action, Observation, Predictor, Termination

if TYPE_CHECKING:
    import gymnasium as gym
    import pyenki


class ActionConfig(ABC):
    """
    The abstract base class of action configurations.

    It defines how a robot actuates actions.
    """

    @property
    @abstractmethod
    def space(self) -> gym.spaces.Box:
        """
        The action space

        Must be implemented by concrete classes!
        """
        ...

    @abstractmethod
    def actuate(self, act: Action, robot: pyenki.DifferentialWheeled,
                dt: float) -> None:
        """
        How a robot actuates an action.

        Must be implemented by concrete classes!

        :param      act:    The action

        :param      robot:  The robot that actuates the action

        :param      dt:     The time step
        """
        ...


class ObservationConfig(ABC):
    """
    The abstract base class of observation configurations.

    It defines how observations are generated from
    the sensor readings and internal state of robots.
    """

    @property
    @abstractmethod
    def space(self) -> gym.spaces.Dict:
        """
        The observation space

        Must be implemented by concrete classes!
        """
        ...

    @abstractmethod
    def get(self, robot: pyenki.DifferentialWheeled) -> Observation:
        """
        How a robot generates an observation.

        :param      robot:  The robot generating the observation

        :returns:   The observation

        Must be implemented by concrete classes!
        """
        ...


@dc.dataclass
class GroupConfig:
    """
    Contains everything required to configure a group of similar agents.
    """
    action: ActionConfig
    """
    Defines how actions are actuated.
    """
    observation: ObservationConfig
    """
    Defines how observations are generated.
    """
    reward: RewardFunction | None = None
    """
    Defines how rewards are assigned.
    If set to ``None``, it will generate a constant -1.
    """
    info: list[InfoFunction] = dc.field(default_factory=list)
    """
    Defines which extra information is added to the info dictionary
    returned by ``reset`` and ``step``.
    """
    terminations: list[Termination] = dc.field(default_factory=list)
    """
    Defines a list of conditions for a robot to terminate an episode,
    which are evaluated in sequence. The first time a returned value
    is different than ``None``, it is recorded as a success (if True)
    or failure (if False) and may cause the episode to terminate
    for the robot or for the whole environment,
    depending on the value of ``terminate_on`` in the constructor
    of :py:class:`enki_env.ParallelEnkiEnv`.
    """

    def get_controller(self,
                       policy: Predictor,
                       deterministic: bool = True,
                       cutoff: float = 0) -> pyenki.Controller:
        """
        Returns a controller, which can be assigned to a robot
        :py:attr:`pyenki.PhysicalObject.control_step_callback`,
        that actuates a policy.

        :param        policy: The policy.

        :param deterministic: Whether to evaluate the policy deterministically.

        :param        cutoff: When the absolute value of actions is below this threshold,
                              they will be set to zero.
        """

        def f(r: pyenki.PhysicalObject, dt: SupportsFloat) -> None:
            robot = cast('pyenki.DifferentialWheeled', r)
            obs = self.observation.get(robot)
            act, _ = policy.predict(obs, deterministic=deterministic)
            if np.all(np.abs(act) < cutoff):
                act *= 0
            self.action.actuate(act, robot, float(dt))

        return f


def make_agents(
    world: pyenki.World, config: dict[str, GroupConfig]
) -> dict[str, tuple[pyenki.DifferentialWheeled, str, GroupConfig]]:
    groups: dict[str, list[pyenki.DifferentialWheeled]] = {
        k: [
            cast('pyenki.DifferentialWheeled', a) for a in world.robots
            if a.name == k or k == ''
        ]
        for k in config
    }
    # uids = {a: f'{k}_{i}' for k, agents in groups.items() for i, a in enumerate(agents)}
    configs = {
        f'{k or "robot"}_{i}': (a, k, config[k])
        for k, agents in groups.items()
        for i, a in enumerate(agents)
    }
    # agent_configs = {a: config[k] for k, agents in groups.items() for a in agents}
    return configs


def setup_controllers(world: pyenki.World,
                      config: dict[str, GroupConfig],
                      policies: dict[str, Predictor],
                      deterministic: bool = True,
                      cutoff: float = 0) -> None:
    """
    Equips all robots in the world, with controllers that evaluate the selected policies,
    by matching the robot name with the keys of ``policies`` and ``config``.

    :param         world: The world

    :param        config: A map of configurations assigned to groups of robots.

    :param      policies: A map of policies assigned to groups of robots.

    :param deterministic: Whether to evaluate the policy deterministically.

    :param        cutoff: When the absolute value of actions is below this threshold,
                          they will be set to zero.
    """
    configs = make_agents(world, config)
    for robot, name, conf in configs.values():
        if name in policies:
            robot.control_step_callback = conf.get_controller(
                policies[name], deterministic=deterministic, cutoff=cutoff)
