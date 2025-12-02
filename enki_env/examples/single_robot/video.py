from __future__ import annotations

import pathlib as pl
from typing import TYPE_CHECKING

import pyenki

from ..utils import video
from .baseline import Baseline
from .environment import make_env
from .rl import get_policy

if TYPE_CHECKING:
    from moviepy import VideoClip


def set_color(r: float, g: float, b: float) -> video.WorldInit:

    def f(world: pyenki.World) -> None:
        for robot in world.robots:
            if isinstance(robot, pyenki.Thymio2):
                robot.set_led_top(r, g, b)

    return f


def make_video() -> VideoClip:
    configs = ((Baseline(), set_color(1, 1,
                                      0)), (get_policy(), set_color(0, 1, 1)))
    return video.make_video(make_env().unwrapped,
                            configs,
                            number=5,
                            duration=3,
                            camera_position=(-25, -20),
                            camera_altitude=10,
                            camera_pitch=-0.3,
                            camera_yaw=0.9)


if __name__ == '__main__':
    import pyenki.viewer
    pyenki.viewer.init()
    path = pl.Path(__file__).parent / "sim.mp4"
    make_video().write_videofile(path, fps=30)
    pyenki.viewer.cleanup()
