from dataclasses import dataclass
from typing import Sequence

import numpy as np
from .vector import Vector, vec2


@dataclass(frozen=True)
class Rect:
    bottom_left: Vector
    top_right: Vector

    @property
    def center(self) -> Vector:
        return np.average(self.bottom_left + self.top_right) #type: ignore
    
    @property
    def dimensions(self) -> Vector:
        return np.abs(self.top_right - self.bottom_left)
    
    @property
    def width(self) -> float:
        return self.dimensions[0]
    

    @property
    def height(self) -> float:
        return self.dimensions[1]
    

    @property
    def corners(self) -> list[Vector]:
        return [self.bottom_left, self.top_right]
    


def get_corners(center: Vector, width: float, height: float) -> tuple[Vector, Vector]:
    dimensions = vec2(width, height)

    return tuple(
        center + sign * dimensions / 2
        for sign in (-1, 1)
    )


def in_bounds(position: Vector, rect: Rect) -> bool:
    return np.all(
        (rect.bottom_left <= position) * (position < rect.top_right)
    ).astype(bool)


def cull(points: Sequence[Vector], rect: Rect) -> Sequence[Vector]:
    return [
        point for point in points if in_bounds(point, rect)
    ]