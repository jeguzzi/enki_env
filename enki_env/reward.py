from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyenki


@dc.dataclass
class ConstReward:
    """
    Generates a constant reward.
    """
    value: float = -1
    """the constant value"""

    def __call__(self, robot: pyenki.DifferentialWheeled,
                 world: pyenki.World) -> float:
        return self.value
