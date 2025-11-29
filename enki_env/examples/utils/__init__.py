import math
import pathlib as pl
from typing import TYPE_CHECKING

from ...types import Predictor

if TYPE_CHECKING:
    import pyenki


def load(path: pl.Path) -> Predictor | None:
    from ...utils.onnx import OnnxPolicy

    path = path.with_suffix(".onnx")
    if path.exists():
        return OnnxPolicy(path)
    return None


def normalize_angle(angle: float) -> float:
    angle = math.fmod(angle, 2 * math.pi)
    if angle > math.pi:
        angle -= 2 * math.pi
    if angle < -math.pi:
        angle += 2 * math.pi
    return angle


def is_still(robot: "pyenki.DifferentialWheeled", speed_tol: float = 1) -> bool:
    return (abs(robot.left_wheel_encoder_speed) < speed_tol
            and abs(robot.right_wheel_encoder_speed) < speed_tol)
