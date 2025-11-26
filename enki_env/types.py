from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np
import numpy.typing
import pyenki

if TYPE_CHECKING:
    import gymnasium as gym
    import torch


class Scenario(Protocol):
    """
    A scenario is a generator of world. It receives a random seed as argument,
    that is should pass to the world constructor.
    Any random sampling should use the :py:attr:`pyenki.World.random_generator`
    to ensure reproducibility.

    For example ::

        def my_scenario(seed: int) -> pyenki.World:
            robot = pyenki.Thymio2(seed=seed)
            robot.angle = world.random_generator.uniform(0, math.pi * 2)
            world.add_object(world)
    """
    def __call__(self, seed: int) -> pyenki.World:
        """
        Creates a world with random seed

        :param seed: the random seed.

        :returns:    the world.
        """
        ...


Array: TypeAlias = numpy.typing.NDArray[np.float64]
Observation: TypeAlias = dict[str, Array]
Action: TypeAlias = Array
Info: TypeAlias = dict[str, Any]
Termination: TypeAlias = Callable[[pyenki.DifferentialWheeled, pyenki.World], bool | None]
State: TypeAlias = tuple[Array, ...]
EpisodeStart: TypeAlias = Array
PathLike: TypeAlias = os.PathLike[str] | str


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

PyTorchObs: TypeAlias = 'torch.Tensor | dict[str, torch.Tensor]'


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
