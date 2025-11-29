from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pyenki


class RewardFunction(Protocol):
    """
    A callable that generates rewards at each step of the
    environment.

    For example ::

        def my_reward(robot: pyenki.DifferentialWheeled, success: bool | None) -> float:
            return -1 if abs(robot.position[0]) > 1 else 0
    """

    def __call__(self, robot: pyenki.DifferentialWheeled,
                 success: bool | None) -> float:
        """
        Generate a reward for a robot.

        :param      robot:    The robot
        :param      success:  Whether the robot has completed
            the task with success (``True``) or failure (``False``), or
            has yet to complete the task or there is no task (``None``).

        :returns:   The reward assigned to the robot
        """
        ...


@dc.dataclass
class ConstReward:
    """
    Generates a constant reward.
    """
    value: float = -1
    """the value in case of undecided success/failure"""
    success_value: float = 0
    """the value in case of success"""
    failure_value: float = -1
    """the value in case of failure"""

    def __call__(self, robot: pyenki.DifferentialWheeled,
                 success: bool | None) -> float:
        return {
            True: self.success_value,
            False: self.failure_value,
            None: self.value
        }[success]
