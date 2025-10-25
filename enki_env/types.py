from __future__ import annotations

from collections.abc import Callable
from typing import Any, SupportsFloat

import numpy as np
import pyenki

Predictor = Any
Scenario = Callable[[np.random.Generator | None], pyenki.World]
Array = np.ndarray
Observation = dict[str, Array]
Action = Array
Info = dict[str, Any]
Termination = Callable[[pyenki.DifferentialWheeled, pyenki.World], bool | None]
Controller = Callable[[pyenki.PhysicalObject, SupportsFloat], None]
