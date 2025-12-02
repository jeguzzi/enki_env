from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from pyenki import World

if TYPE_CHECKING:
    from moviepy import VideoClip

WorldInit = Callable[[World], None]
PolicyWithInit = tuple[Any, WorldInit]


def make_video(env: Any,
               configs: Sequence[PolicyWithInit],
               number: int = 5,
               time_step: float = 0.03,
               width: int = 1280,
               height: int = 720,
               **kwargs: Any) -> VideoClip:
    from moviepy import concatenate_videoclips
    from pyenki.video import make_video

    videos = []
    for seed in range(number):
        for policy, init in configs:
            world = env.make_world(policy, seed=seed, cutoff=0.0)
            init(world)
            videos.append(
                make_video(world,
                           time_step=time_step,
                           width=width,
                           height=height,
                           **kwargs))
    return concatenate_videoclips(videos)
