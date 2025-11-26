from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyenki import DifferentialWheeled, World


@dc.dataclass
class ConstReward:
    """
    Generates a constant reward.
    """
    value: float = -1
    """the constant value"""
    def __call__(self, robot: DifferentialWheeled, world: World) -> float:
        return self.value
