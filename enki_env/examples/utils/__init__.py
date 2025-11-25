import pathlib as pl

from ...types import Predictor


def load(path: pl.Path) -> Predictor | None:
    from ...utils.onnx import OnnxPolicy

    path = path.with_suffix(".onnx")
    if path.exists():
        return OnnxPolicy(path)
    return None
