from __future__ import annotations

import pathlib as pl
from typing import TYPE_CHECKING

from ..single_robot.video import set_color
from ..utils import video
from .baseline import Baseline
from .environment import make_env
from .rl import get_policy

if TYPE_CHECKING:
    from moviepy import VideoClip


def make_video() -> VideoClip:
    configs = (({
        'thymio': Baseline()
    }, set_color(1, 1, 0)), ({
        'thymio': get_policy()
    }, set_color(0, 1, 1)))
    return video.make_video(make_env(),
                            configs,
                            number=5,
                            duration=3,
                            camera_position=(10, -30),
                            camera_altitude=15,
                            camera_pitch=-0.5,
                            camera_yaw=1.57)


if __name__ == '__main__':
    import pyenki.viewer
    pyenki.viewer.init()
    path = pl.Path(__file__).parent / "sim.mp4"
    make_video().write_videofile(path, fps=30)
    pyenki.viewer.cleanup()
