import numpy as np
from typing import Iterable, Union, Self, Callable, Sequence
from dataclasses import dataclass, field

from .particle import Particle
from .vector import vec2, Vector
from . import rect


def subdivide(points: Iterable[Vector], center):
    """
    Subdivide into four quadrants, starting with the top-right, going anticlockwise (Mathematical / Cartesian convention)
    """

    quadrants = [[], [], [], []]

    comparisons = [(point >= center) for point in points]

    for particle, greater_equal in zip(points, comparisons):
        if np.all(greater_equal):
            quadrants[0].append(particle)

        elif not greater_equal[0] and greater_equal[1]:
            quadrants[1].append(particle)

        elif not greater_equal[0] and not greater_equal[1]:
            quadrants[2].append(particle)

        else:
            quadrants[3].append(particle)

    return quadrants


@dataclass(frozen=True)
class Quadtree:
    points: Sequence[Vector]
    rect: rect.Rect
    particle_masses: Sequence[float]
    parent_node: Self | None = None

    child_nodes: list[Self] = field(init=False)
    mass: float = field(init=False)
    position: Vector = field(init=False)
    diameter: float = field(init=False)

    def __post_init__(self):
        for point in self.points:
            if not rect.in_bounds(point, self.rect):
                raise ValueError("All particles must be within the Quadtree")

        if len(self.points) <= 1:
            object.__setattr__(self, "child_nodes", None)
            return

        child_dimensions = self.rect.dimensions / 2

        child_centers = [
            self.rect.center + child_dimensions / 2 * vec2(*signs)
            for signs in [(1, 1), (-1, 1), (-1, -1), (1, -1)]
        ]

        child_centers = []

        child_quadrants = subdivide(self.points, self.rect.center)

        child_nodes = [
            Quadtree(
                point_array,
                rect.Rect(*rect.get_corners(center, *child_dimensions)),
                self.particle_masses,
                self,
            )
            for point_array, center in zip(child_quadrants, child_centers)
        ]

        object.__setattr__(self, "child_nodes", child_nodes)

        object.__setattr__(
            self, "mass", sum(self.particle_masses)
        )
        object.__setattr__(self, "position", self.rect.center)
        object.__setattr__(self, "diameter", np.average(self.rect.dimensions))
