from __future__ import annotations

import dataclasses as dc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, SupportsFloat, cast

from .types import (Action, InfoFunction, Observation, Predictor,
                    RewardFunction, Termination)

if TYPE_CHECKING:
    import gymnasium as gym
    from pyenki import Controller, DifferentialWheeled, PhysicalObject, World


class ActionConfig(ABC):
    """
    The abstract base class of action configurations.

    It defines how actions are actuated.
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
    def actuate(self, act: Action, robot: DifferentialWheeled,
                dt: float) -> None:
        """
        Make a robot actuates an action.

        Must be implemented by concrete classes!

        :param      act:    The action
        :param      robot:  The robot that actuates the action
        :param      dt:     The time step
        """
        ...


class ObservationConfig(ABC):
    """
    The abstract base class of observation configurations.

    It defines how observations are generated from robots'
    sensor readings and internal state.
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
    def get(self, robot: DifferentialWheeled) -> Observation:
        """
        Makes a robot generate an observation.

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
    Defines how reward are assigned.
    If set to ``None``, it will generate a constant -1.
    """
    info: InfoFunction | None = None
    """
    Defines which extra information is generated.
    If set to ``None``, it will generate an empty dictionary
    """
    terminations: list[Termination] = dc.field(default_factory=list)
    """
    Defines a list of conditions for a robot to terminate an episode.
    Condition are evaluated in sequence. The first returned value
    different than ``None``, it is assigned as a ``success`` and
    may make the episode terminate for the agent or for the whole group,
    depending on the value of ``terminate_on`` in the constructor
    of :py:class:`enki_env.ParallelEnkiEnv`.
    """
    def get_controller(self, policy: Predictor) -> Controller:
        """
        Returns a controller, which can be assigned to a robot's
        :py:attr:`pyenki.PhysicalObject.control_step_callback`,
        that actuates a policy.
        """
        def f(r: PhysicalObject, dt: SupportsFloat) -> None:
            robot = cast('DifferentialWheeled', r)
            obs = self.observation.get(robot)
            act, _ = policy.predict(obs)
            self.action.actuate(act, robot, float(dt))

        return f


def make_agents(
    world: World, config: dict[str, GroupConfig]
) -> dict[str, tuple[DifferentialWheeled, str, GroupConfig]]:
    groups: dict[str, list[DifferentialWheeled]] = {
        k: [
            cast('DifferentialWheeled', a) for a in world.robots
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


def setup_controllers(world: World, config: dict[str, GroupConfig],
                      policies: dict[str, Predictor]) -> None:
    """
    Equip robots in the world with controllers that evaluate
    the given policies.

    :param      world:     The world
    :param      config:    A map of groups configurations assigned to (group) a name
    :param      policies:  A map policies assigned to (group) name
    """
    configs = make_agents(world, config)
    for robot, name, conf in configs.values():
        if name in policies:
            robot.control_step_callback = conf.get_controller(policies[name])
