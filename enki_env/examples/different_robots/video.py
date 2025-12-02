from __future__ import annotations

import pathlib as pl
from typing import TYPE_CHECKING, Any

from ..single_robot.video import set_color
from ..utils import video
from .baseline import ThymioBaseline  # type: ignore[attr-defined]
from .baseline import EPuckBaseline
from .environment import make_env
from .rl import get_policies
from .centralized_policy_rl import get_policy
from ...concat_env import ConcatEnv

if TYPE_CHECKING:
    from moviepy import VideoClip


def make_video(centralized: bool = False) -> VideoClip:

    env: Any = make_env()
    if centralized:
        env = ConcatEnv(env)
        configs: Any = ((get_policy(), set_color(1, 0, 1)), )
    else:
        configs = (({
            'thymio': ThymioBaseline(),
            'e-puck': EPuckBaseline()
        }, set_color(1, 1, 0)), (get_policies(), set_color(0, 1, 1)))
    return video.make_video(env,
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
    path = pl.Path(__file__).parent / "centralized_policy_sim.mp4"
    make_video(True).write_videofile(path, fps=30)
    pyenki.viewer.cleanup()
