from __future__ import annotations

import pathlib as pl

import pyenki

from ..utils.video import WorldInit, make_video
from .environment import make_env
from .baseline import Baseline
from .rl import get_policy


def set_color(r: float, g: float, b: float) -> WorldInit:

    def f(world: pyenki.World) -> None:
        for robot in world.robots:
            if isinstance(robot, pyenki.Thymio2):
                robot.set_led_top(r, g, b)

    return f


if __name__ == '__main__':
    import pyenki.viewer
    pyenki.viewer.init()

    configs = ((Baseline(), set_color(1, 1,
                                    0)), (get_policy(), set_color(0, 1, 1)))
    video = make_video(make_env(),
                       configs,
                       number=5,
                       duration=3,
                       camera_position=(-25, -20),
                       camera_altitude=10,
                       camera_pitch=-0.3,
                       camera_yaw=0.9)
    path = pl.Path(__file__).parent / "sim.mp4"
    video.write_videofile(path, fps=30)
    pyenki.viewer.cleanup()
