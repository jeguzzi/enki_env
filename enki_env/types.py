from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np
import numpy.typing

if TYPE_CHECKING:
    import gymnasium as gym
    import pyenki
    import torch

Array: TypeAlias = numpy.typing.NDArray[np.float64]
BoolArray: TypeAlias = numpy.typing.NDArray[np.bool_]
Observation: TypeAlias = dict[str, Array]
Action: TypeAlias = Array
Info: TypeAlias = dict[str, Any]
State: TypeAlias = tuple[Array, ...]
EpisodeStart: TypeAlias = Array
PathLike: TypeAlias = os.PathLike[str] | str
PyTorchObs: TypeAlias = 'torch.Tensor | dict[str, torch.Tensor]'


class RewardFunction(Protocol):
    """
    A callable that generates rewards at each step of the
    environment.

    For example ::

        def my_reward(robot: pyenki.DifferentialWheeled, world: pyenki.World) -> float:
            return -1 if abs(robot.position[0]) > 1 else 0
    """

    def __call__(self, robot: pyenki.DifferentialWheeled,
                 world: pyenki.World) -> float:
        """
        Generate a reward for a robot.

        :param      robot:  The robot
        :param      world:  The world the robot belongs to.

        :returns:   The reward assigned to the robot
        """
        ...


class InfoFunction(Protocol):
    """
    A callable that generates extra information at each step of the
    environment.
    """

    def __call__(self, robot: pyenki.DifferentialWheeled,
                 world: pyenki.World) -> Info:
        """
        Generate information related to a robot

        :param      robot:  The robot
        :param      world:  The world the robot belongs to.

        :returns:   The information related to the robot
        """
        ...


class Termination(Protocol):
    """
    A criterion to decide the success/failure of an episode for a given robot.
    Should return ``True`` for success, ``False`` for failure, and ``None`` if not yet decided.

    For example, for a task where a robot needs to travel along the positive x-direction,
    we may select failure when if exits some narrow bands and
    success when it travels further enough:

    .. code:: Python

       def my_criterion(robot: pyenki.DifferentialWheeled,
                        world: pyenki.World) -> bool | None:
           if robot.position[1] > 100:
               return True
           if abs(robot.position[0]) > 10:
               return False
           return None
    """

    def __call__(self, robot: pyenki.DifferentialWheeled,
                 world: pyenki.World) -> bool | None:
        """
        Decides if the episode should terminate for a given robot

        :param      robot:  The robot
        :param      world:  The world the robot belongs to.

        :returns:   ``True`` to terminate with success,
                    ``False`` to terminate with failure,
                    ``None`` to not terminate.
        """
        ...


class Predictor(Protocol):
    """
    This class describes the predictor protocol.

    Same as :py:type:`stable_baselines3.common.type_aliases.PolicyPredictor`,
    included here to be self-contained.
    """

    @property
    def action_space(self) -> gym.Space[Any]:
        ...

    @property
    def observation_space(self) -> gym.Space[Any]:
        ...

    def predict(self,
                observation: Observation,
                state: State | None = None,
                episode_start: EpisodeStart | None = None,
                deterministic: bool = False) -> tuple[Action, State | None]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        ...


class PyTorchPolicy(Predictor, Protocol):

    def __call__(self,
                 obs: PyTorchObs,
                 deterministic: bool = False) -> torch.Tensor:
        """
        Evaluate the policy

        :param      obs:            The observations
        :param      deterministic:  Whether or not to return deterministic actions.
        """
        ...

    def forward(self,
                obs: PyTorchObs,
                deterministic: bool = False) -> torch.Tensor:
        """
        Evaluate the policy

        :param      obs:            The observations
        :param      deterministic:  Whether or not to return deterministic actions.
        """
        ...
