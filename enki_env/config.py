from __future__ import annotations

import dataclasses as dc
from abc import ABC, abstractmethod
from typing import Any, SupportsFloat, cast, TYPE_CHECKING


from .types import Action, Controller, Observation, Predictor, Termination

if TYPE_CHECKING:
    from pyenki import DifferentialWheeled, PhysicalObject, World
    import gymnasium as gym


class ActionConfig(ABC):

    @property
    @abstractmethod
    def space(self) -> gym.spaces.Box:
        ...

    @abstractmethod
    def actuate(self, act: Action, robot: DifferentialWheeled,
                dt: float) -> None:
        ...


class ObservationConfig(ABC):

    @property
    @abstractmethod
    def space(self) -> gym.spaces.Dict:
        ...

    @abstractmethod
    def get(self, robot: DifferentialWheeled) -> Observation:
        ...


class RewardConfig(ABC):

    @abstractmethod
    def get(self, robot: DifferentialWheeled, world: World) -> float:
        ...


class InfoConfig(ABC):

    @abstractmethod
    def get(self, robot: DifferentialWheeled, world: World) -> dict[str, Any]:
        ...


class EmptyInfoConfig(InfoConfig):

    def get(self, robot: DifferentialWheeled, world: World) -> dict[str, Any]:
        return {}


class ConstRewardConfig(RewardConfig):

    def get(self, robot: DifferentialWheeled, world: World) -> float:
        return -1


@dc.dataclass
class GroupConfig:
    action: ActionConfig
    observation: ObservationConfig
    reward: RewardConfig = ConstRewardConfig()
    info: InfoConfig = EmptyInfoConfig()
    terminations: list[Termination] = dc.field(default_factory=list)

    def get_control(self, policy: Predictor) -> Controller:

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


def setup_policies(world: World, config: dict[str, GroupConfig],
                   policies: dict[str, Predictor]) -> None:
    configs = make_agents(world, config)
    for robot, name, conf in configs.values():
        if name in policies:
            robot.control_step_callback = conf.get_control(policies[name])
