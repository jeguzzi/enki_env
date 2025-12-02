from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, SupportsFloat, cast

import gymnasium as gym
import numpy as np

from .config import make_agents
from .env import BaseEnv
from .parallel_env import ParallelEnkiEnv
from .rollout import Rollout
from .types import Action, Array, Info, Observation, Predictor

if TYPE_CHECKING:
    import pyenki

    # from enki_env.parallel_env import BaseParallelEnv


def concat_infos(values: dict[str, dict[str, Array]]) -> dict[str, Array]:
    vs = list(values.values())
    keys = set.union(*[set(value) for value in vs])
    return {
        k:
        np.concatenate([np.atleast_1d(value[k]) for value in vs if k in value])
        for k in keys
    }


def concat_obs(values: dict[str, dict[str, Array]],
               space: gym.spaces.Dict) -> dict[str, Array]:
    vs = list(values.values())
    keys = set(space)
    return {
        k: np.concatenate([value[k].flatten() for value in vs if k in value])
        for k in keys
    }


def split_actions(values: Array,
                  spaces: dict[str, gym.spaces.Box]) -> dict[str, Array]:
    vs = {}
    i = 0
    for name, space in spaces.items():
        j = i + int(np.prod(space.shape))
        vs[name] = values[i:j].reshape(space.shape)
        i = j
    return vs


def concat_box_spaces(spaces: list[gym.spaces.Box]) -> gym.spaces.Box:
    low = np.concatenate([space.low.flatten() for space in spaces])
    high = np.concatenate([space.high.flatten() for space in spaces])
    return gym.spaces.Box(low, high)


def concat_dict_spaces(spaces: list[gym.spaces.Dict]) -> gym.spaces.Dict:
    keys = set.union(*[set(space) for space in spaces])
    return gym.spaces.Dict({
        key:
        concat_box_spaces([
            cast('gym.spaces.Box', space[key]) for space in spaces
            if key in space.keys()
        ])
        for key in keys
    })


class ConcatEnv(gym.Env[Observation, Action]):
    """
    Wraps a multi-robot environment as a single
    agent environment, concatenating observations and
    information dictionaries and aggregating
    rewards, terminations and truncations.

    :param env:  The parallel environment
    """

    def __init__(self, env: ParallelEnkiEnv) -> None:
        if len(env.possible_agents) > 1 and env._terminate_on is None:
            raise ValueError(
                "Requires an environment with termination that is synchronized."
                "terminate_on should be set to 'any' or 'all'")
        self._reduce = all if env._terminate_on == 'all' else any
        self.env = env
        self.use_state = False
        self.observation_space: gym.spaces.Dict = concat_dict_spaces(
            list(
                cast('dict[str, gym.spaces.Dict]',
                     env.observation_spaces).values()))
        self.action_space: gym.spaces.Box = concat_box_spaces(
            list(
                cast('dict[str, gym.spaces.Box]', env.action_spaces).values()))

    @property
    def unwrapped(self) -> BaseEnv:
        return self.env.unwrapped  # type: ignore

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """
        Conforms to :py:meth:`gymnasium.Env.reset`.

        Resets the parallel environment and return concatenated
        observation and infos.
        """
        obs, infos = self.env.reset(seed=seed, options=options)
        return concat_obs(obs, self.observation_space), concat_infos(infos)

    def step(self,
             action: Action) -> tuple[Observation, float, bool, bool, Info]:
        """
        Conforms to :py:meth:`gymnasium.Env.step`.

        Converts the action to an agent-indexed dictionary
        ``action`` -> ``{0: act[0:...], 1: act[...:...], ...}``
        and forwards them to :py:meth:`pettingzoo.utils.env.ParallelEnv.step`.

        Returns concatenated observations
        and infos, and aggregated reward (sum), termination (all) , truncation (any).
        """
        act = split_actions(
            action, cast('dict[str, gym.spaces.Box]', self.env.action_spaces))
        obs, reward, terminated, truncated, infos = self.env.step(act)
        return (concat_obs(obs, self.observation_space),
                sum(x for _, x in reward.items()),
                self._reduce(x for _, x in terminated.items()),
                any(x for _, x in truncated.items()), concat_infos(infos))

    def display_in_notebook(self) -> None:
        """
        Displays the environment in a notebook using a
        an interactive :py:class:`pyenki.buffer.EnkiRemoteFrameBuffer`.

        Requires ``render_mode="human"`` and a notebook.
        """
        return self.env.display_in_notebook()

    def _get_policy(
        self,
        policy: Predictor | None = None,
        deterministic: bool = True,
        cutoff: float = 0
    ) -> Callable[[dict[str, Observation]], dict[str, Action]]:

        def f(obs: dict[str, Observation]) -> dict[str, Action]:
            if policy:
                observation = concat_obs(obs, self.observation_space)
                action = policy.predict(observation,
                                        deterministic=deterministic)[0]
                actions = split_actions(
                    action,
                    cast('dict[str, gym.spaces.Box]', self.env.action_spaces))
                for k in actions:
                    if np.all(np.abs(actions[k]) < cutoff):
                        actions[k] *= 0
                return actions
            else:
                return {
                    k: space.sample()
                    for k, space in self.env.action_spaces.items()
                }

        return f

    def rollout(self,
                policy: Predictor | None = None,
                max_steps: int = -1,
                seed: int = 0,
                deterministic: bool = True,
                cutoff: float = 0) -> dict[str, Rollout]:
        """
        Performs a rollout of an episode

        :param      policy:     The policy to apply; if not provided,
            it will randomly generate actions.
        :param      max_steps:  The maximum number of steps to perform.
        :param      seed:       The random seed.
        :param      deterministic: Whether to evaluate the policies
            deterministically.
        :param cutoff: When the absolute value of actions is below this threshold,
            they will be set to zero.

        :returns:   A dictionary, keyed by group, with
            the data collected during the rollout.
        """
        return self.env._rollout(self._get_policy(policy,
                                                  deterministic=deterministic,
                                                  cutoff=cutoff),
                                 max_steps=max_steps,
                                 seed=seed)

    def get_controller(
            self,
            world: pyenki.World,
            policy: Predictor | None = None,
            deterministic: bool = True,
            cutoff: float = 0
    ) -> Callable[[pyenki.World, SupportsFloat], None]:
        configs = make_agents(world, self.env.config)
        p = self._get_policy(policy,
                             deterministic=deterministic,
                             cutoff=cutoff)

        def f(world: pyenki.World, dt: SupportsFloat) -> None:
            obs = {
                name: conf.observation.get(robot)
                for name, (robot, _, conf) in configs.items()
            }
            acts = p(obs)
            for name, (robot, _, config) in configs.items():
                config.action.actuate(acts[name], robot, float(dt))

        return f

    def make_world(self,
                   policy: Predictor | None = None,
                   seed: int = 0,
                   deterministic: bool = True,
                   cutoff: float = 0) -> pyenki.World:
        """
        Generates a world using the scenario and
        assign a centralized policy to the world controller.

        :param      policy:     The centralized policy to apply; if not provided,
            it will randomly generate actions.
        :param      seed:      The random seed.
        :param      deterministic: Whether to evaluate the policy
            deterministically.
        :param cutoff: When the absolute value of actions is below this threshold,
            they will be set to zero.

        :returns:   The world
        """
        world = self.env._scenario(seed)
        world.control_step_callback = self.get_controller(
            world, policy, deterministic=deterministic, cutoff=cutoff)
        return world
