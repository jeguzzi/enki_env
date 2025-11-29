from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from .types import Info

if TYPE_CHECKING:
    import pyenki


class InfoFunction(Protocol):
    """
    A callable that generates extra information at each step of the
    environment.
    """

    def __call__(self, robot: pyenki.DifferentialWheeled, info: Info) -> None:
        """
        Adds information related to a robot.

        :param      robot:  The robot.
        :param      info:   The information dictionary.
        """
        ...


def pose_info(robot: pyenki.DifferentialWheeled, info: Info) -> None:
    """
    Adds the robot position in key ``"position"`` and the robot orientation in key ``"angle"``.
    """
    info.update(position=robot.position, angle=np.asarray(robot.angle))


def twist_info(robot: pyenki.DifferentialWheeled, info: Info) -> None:
    """
    Adds the robot velocity in key ``"velocity"`` and the angular speed
    in key ``"angular_speed"``.
    """
    info.update(velocity=robot.velocity,
                angular_speed=np.asarray(robot.angular_speed))


def wheel_info(robot: pyenki.DifferentialWheeled, info: Info) -> None:
    """
    Adds the ``[left, right]`` wheel speeds in key ``"wheel_speeds"``.
    """
    info.update(wheel_speeds=np.asarray(
        [robot.left_wheel_encoder_speed, robot.right_wheel_encoder_speed]))
