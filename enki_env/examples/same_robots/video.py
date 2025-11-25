from __future__ import annotations

import pathlib as pl

from ..single_robot.video import set_color
from ..utils.video import make_video
from .baseline import Baseline
from .environment import make_env
from .rl import get_policy

if __name__ == '__main__':
    import pyenki.viewer
    pyenki.viewer.init()

    configs = (({
        '': Baseline()
    }, set_color(1, 1, 0)), ({
        '': get_policy()
    }, set_color(0, 1, 1)))
    video = make_video(make_env(),
                       configs,
                       number=5,
                       duration=3,
                       camera_position=(10, -30),
                       camera_altitude=15,
                       camera_pitch=-0.5,
                       camera_yaw=1.57)
    path = pl.Path(__file__).parent / "sim.mp4"
    video.write_videofile(path, fps=30)
    pyenki.viewer.cleanup()
