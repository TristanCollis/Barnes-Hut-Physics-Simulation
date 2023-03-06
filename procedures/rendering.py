from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image as img

from structs import Vector, vec2, Rect, in_bounds

Animation = Sequence[img.Image]
Frame = NDArray[np.uint8]


def animation_from_positions(position_history: Sequence[Vector], window: Rect, resolution: Sequence[int]) -> Animation:
    return render_animation(
        [
            create_frame(positions, window, resolution)
            for positions in position_history
        ]
    )


def render_frame(frame: Frame) -> img.Image:
    return img.fromarray(np.rot90(frame), mode="L")


def render_animation(frames: Sequence[Frame]) -> Animation:
    return [render_frame(frame) for frame in frames]


def save_animation(animation: Animation, out_path: str) -> None:
    path = out_path

    if not out_path.endswith(".gif"):
        path += ".gif"

    animation[0].save(
        path,
        save_all=True,
        append_images=animation[1:],
        loop=0,
        #duration=loop_duration * 1000 / len(animation),
        duration=20
    )


def create_frame(
    positions: Iterable[Vector],
    window: Rect,
    resolution: Sequence[int],
) -> Frame:
    frame_array = np.zeros(resolution, dtype=np.uint8) + 10

    transformed_positions = positions - window.bottom_left
    transformed_window = Rect(vec2(0, 0), window.dimensions)

    for position in transformed_positions:
        if not in_bounds(position, transformed_window):
            continue

        scaled_coord = np.floor(
            position * resolution / transformed_window.top_right
        ).astype(int)
        frame_array[*scaled_coord] = 255

    return frame_array.astype(np.uint8)
