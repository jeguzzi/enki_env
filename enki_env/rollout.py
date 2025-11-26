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
    Data collected during a rollout like in
    :py:meth:`pyenki.ParallelEnkiEnv.rollout` and
    :py:meth:`pyenki.EnkiEnv.rollout`.
    """
    observation: Observation
    """
    An dictionary of observation values for each key of the observation space.
    Array have shape ``(#steps, #agents, *shape of field)`` and store
    the value of the field at each step for each agents.
    Validity in stored separately in ``valid``.
    """
    action: Action
    """
    An array of shape ``(#steps, #agents, *shape of actions)`` with the actions at each step
    for each agents. Validity in stored separately in ``valid``.
    """
    reward: Array
    """
    An array of shape ``(#steps, #agents)`` with the rewards at each step
    for each agents. Validity in stored separately in ``valid``.
    """
    termination: numpy.typing.NDArray[np.bool_]
    """
    An array of length #agents that stores whether the agents were terminated.
    """
    truncation: numpy.typing.NDArray[np.bool_]
    """
    An array of length #agents that stores whether the agents where truncated
    """
    info: dict[str, numpy.ma.masked_array]
    """
    A dictionary where the value is a masked array
    of shape ``(steps, agents, *shape of entry)``
    aggregates all the information entries
    for the given key for all agents and all steps.
    The array mask selects entries that where present
    at time t for agent a.
    """
    valid: numpy.typing.NDArray[np.bool_]
    """
    ``(#steps + 1, #agents)`` array of Booleans where
    ``valid(i, j) == True`` iff agent j was alive at time i.
    """
    agents: list[str]
    """
    The name of the agents, in the same order as in the data fields.
    """

    @property
    def masked_action(self) -> numpy.ma.masked_array:
        """
        Returns the actions as masked array

        :returns:   The actions
        """
        size = numpy.prod(self.action.shape[2:])
        valid = self.valid[:-1, :].repeat(size).reshape(self.action.shape)
        return numpy.ma.masked_array(self.action,  # type: ignore[no-untyped-call]
                                     ~valid)

    @property
    def masked_observation(self) -> dict[str, numpy.ma.masked_array]:
        """
        Returns the observations as dictionary of masked array

        :returns:   The observations
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
        Returns the rewards as masked array

        :returns:   The rewards
        """
        return numpy.ma.masked_array(  # type: ignore[no-untyped-call]
            self.reward, ~self.valid[1:, :])

    @property
    def episode_reward(self) -> float:
        """
        Returns the average (over agents) cumulative (over steps) reward

        :returns:   The episode reward
        """
        return self.masked_reward.sum(axis=0).mean()  # type: ignore

    @property
    def episode_length(self) -> int:
        """
        The number of steps

        :returns:   The steps
        """
        return len(self.reward)

    @property
    def length(self) -> numpy.typing.NDArray[np.int_]:
        """
        The number of steps for each agent

        :returns:   An array of integers of length #agents
        """
        return self.valid.sum(axis=0) - 1  # type: ignore

    @staticmethod
    def aggregate(agents: Sequence[str], action_space: gym.spaces.Box,
                  observation_space: gym.spaces.Dict,
                  actions: Sequence[dict[str, Action]],
                  observations: Sequence[dict[str, Observation]],
                  rewards: Sequence[dict[str, float]],
                  terminations: Sequence[dict[str, bool]],
                  truncations: Sequence[dict[str, bool]],
                  infos: Sequence[dict[str, Info]]) -> Rollout:
        valid = np.stack([[agent in o for agent in agents]
                          for o in observations])
        void_action = action_space.sample() * 0
        action = np.stack([[a.get(agent, void_action) for agent in agents]
                           for a in actions])
        keys = set(observation_space.keys())
        void_observation = {
            k: v * 0
            for k, v in observation_space.sample().items()
        }
        observation = {
            k:
            np.stack([[o.get(agent, void_observation)[k] for agent in agents]
                      for o in observations])
            for k in keys
        }
        reward = np.stack([[r.get(agent, 0) for agent in agents]
                           for r in rewards])
        terms = ChainMap(*terminations[::-1])
        termination = np.array([terms.get(agent, False) for agent in agents])
        truncs = ChainMap(*truncations[::-1])
        truncation = np.array([truncs.get(agent, False) for agent in agents])
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
                [[i.get(agent, {}).get(key, value) for agent in agents]
                 for i in infos])
            valid_info = np.stack([[
                valid_entry if
                (agent in i and key in i[agent]) else invalid_entry
                for agent in agents
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
                       agents=list(agents))
