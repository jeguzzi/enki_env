from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .types import Info

if TYPE_CHECKING:
    import pyenki


def pose_info(robot: pyenki.DifferentialWheeled, world: pyenki.World) -> Info:
    """
    Returns an information dictionary with the robot position
    in key ``"position"`` and the robot orientation in key ``"angle"``.
    """
    return {'position': robot.position, 'angle': np.asarray(robot.angle)}
