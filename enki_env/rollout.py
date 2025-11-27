from __future__ import annotations

import dataclasses as dc
from collections import ChainMap
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.ma
import numpy.typing

from .types import Action, Array, Info, Observation

if TYPE_CHECKING:
    import gymnasium as gym


@dc.dataclass
class Rollout:
    """
    Holds data collected during a rollout, like by
    :py:meth:`enki_env.ParallelEnkiEnv.rollout` and
    :py:meth:`enki_env.EnkiEnv.rollout`.
    """
    observation: Observation
    """
    A dictionary of arrays for each key in the observation space.
    Array have shape ``(#steps, #robots, *shape of observation)`` with
    the value of the observation at each step for each robots.
    Validity in stored separately in ``valid``.
    """
    action: Action
    """
    An array of shape ``(#steps, #robots, *shape of actions)`` with the actions at each step
    for each robots. Validity in stored separately in ``valid``.
    """
    reward: Array
    """
    An array of shape ``(#steps, #robots)`` with the rewards at each step
    for each robots. Validity in stored separately in ``valid``.
    """
    termination: numpy.typing.NDArray[np.bool_]
    """
    An array of length ``#robots`` that stores whether the robots were terminated.
    """
    truncation: numpy.typing.NDArray[np.bool_]
    """
    An array of length ``#robots`` that stores whether the robots were truncated
    """
    info: dict[str, numpy.ma.masked_array]
    """
    A dictionary where the value is a masked array
    of shape ``(#steps, #robots, *shape of info)``
    aggregates all the information entries
    for the given key at each step for all robots.
    The array mask record which entries were present
    at a given time step for a given robot.
    """
    valid: numpy.typing.NDArray[np.bool_]
    """
    An array of shape ``(#steps + 1, #robots)`` of Booleans where
    ``valid(i, j) == True`` iff robot j was alive at time i.
    """
    robots: list[str]
    """
    The name of the robots in the same order as the data fields.
    """

    @property
    def masked_action(self) -> numpy.ma.masked_array:
        """
        Returns the actions as masked array.

        :returns:   The actions.
        """
        size = numpy.prod(self.action.shape[2:])
        valid = self.valid[:-1, :].repeat(size).reshape(self.action.shape)
        return numpy.ma.masked_array(
            self.action,  # type: ignore[no-untyped-call]
            ~valid)

    @property
    def masked_observation(self) -> dict[str, numpy.ma.masked_array]:
        """
        Returns the observations as dictionary of masked array.

        :returns:   The observations.
        """
        return {
            k:
            numpy.ma.masked_array(  # type: ignore[no-untyped-call]
                v, ~self.valid.repeat(np.prod(v.shape[2:])).reshape(v.shape))
            for k, v in self.observation.items()
        }

    @property
    def masked_reward(self) -> numpy.ma.masked_array:
        """
        Returns the rewards as masked array.

        :returns:   The rewards.
        """
        return numpy.ma.masked_array(  # type: ignore[no-untyped-call]
            self.reward, ~self.valid[1:, :])

    @property
    def episode_reward(self) -> float:
        """
        Returns the average (over robots) of cumulative (over steps) rewards.

        :returns:   The episode reward.
        """
        return self.masked_reward.sum(axis=0).mean()  # type: ignore

    @property
    def episode_length(self) -> int:
        """
        The number of steps

        :returns:   The steps.
        """
        return len(self.reward)

    @property
    def episode_success(self) -> numpy.typing.NDArray[np.bool_] | None:
        """
        The final ``info["is_success"]`` for each robot.

        :returns:   An array of Booleans.
        """
        if not 'is_success' in self.info:
            return None
        s = self.info['is_success']
        return s[~s.mask].astype(bool)

    @property
    def length(self) -> numpy.typing.NDArray[np.int_]:
        """
        The number of steps done by each robot.

        :returns:   An array of integers of length ``#robots``
        """
        return self.valid.sum(axis=0) - 1  # type: ignore

    @staticmethod
    def aggregate(robots: Sequence[str], action_space: gym.spaces.Box,
                  observation_space: gym.spaces.Dict,
                  actions: Sequence[dict[str, Action]],
                  observations: Sequence[dict[str, Observation]],
                  rewards: Sequence[dict[str, float]],
                  terminations: Sequence[dict[str, bool]],
                  truncations: Sequence[dict[str, bool]],
                  infos: Sequence[dict[str, Info]]) -> Rollout:
        valid = np.stack([[robot in o for robot in robots]
                          for o in observations])
        void_action = action_space.sample() * 0
        action = np.stack([[a.get(robot, void_action) for robot in robots]
                           for a in actions])
        keys = set(observation_space.keys())
        void_observation = {
            k: v * 0
            for k, v in observation_space.sample().items()
        }
        observation = {
            k:
            np.stack([[o.get(robot, void_observation)[k] for robot in robots]
                      for o in observations])
            for k in keys
        }
        reward = np.stack([[r.get(robot, 0) for robot in robots]
                           for r in rewards])
        terms = ChainMap(*terminations[::-1])
        termination = np.array([terms.get(robot, False) for robot in robots])
        truncs = ChainMap(*truncations[::-1])
        truncation = np.array([truncs.get(robot, False) for robot in robots])
        zero_info = {}
        for i in infos:
            for _, vs in i.items():
                for k, v in vs.items():
                    if k not in zero_info and isinstance(v, np.ndarray):
                        zero_info[k] = v * 0
        info: dict[str, numpy.ma.masked_array] = {}

        for key, value in zero_info.items():
            valid_entry = np.full(zero_info[key].shape, True)
            invalid_entry = np.full(zero_info[key].shape, False)
            data = np.stack(
                [[i.get(robot, {}).get(key, value) for robot in robots]
                 for i in infos])
            valid_info = np.stack([[
                valid_entry if
                (robot in i and key in i[robot]) else invalid_entry
                for robot in robots
            ] for i in infos])
            info[key] = numpy.ma.masked_array(  # type: ignore[no-untyped-call]
                data, ~valid_info)
        return Rollout(observation=observation,
                       reward=reward,
                       termination=termination,
                       truncation=truncation,
                       action=action,
                       info=info,
                       valid=valid,
                       robots=list(robots))
