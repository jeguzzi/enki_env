import pathlib as pl
import math

from ...types import Predictor


def load(path: pl.Path) -> Predictor | None:
    from ...utils.onnx import OnnxPolicy

    path = path.with_suffix(".onnx")
    if path.exists():
        return OnnxPolicy(path)
    return None


def normalize_angle(angle: float) -> float:
    return math.fmod(angle + math.pi, 2 * math.pi) - math.pi
