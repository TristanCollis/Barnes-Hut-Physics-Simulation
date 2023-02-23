import numpy as np
from typing import Iterable, Union
from dataclasses import dataclass, field

from particle import Particle
from vector import Vector2


def subdivide(center, particles: Iterable[Particle]):
    """
    Subdivide into four quadrants, starting with the top-right, going anticlockwise (Mathematical / Cartesian convention)
    """

    quadrants = [[], [], [], []]

    comparisons = [(particle, Vector2(particle.position >= center))
                   for particle in particles]

    for particle, greater_equal in comparisons:

        if greater_equal.x and greater_equal.y:
            quadrants[0].append(particle)

        elif not greater_equal.x and greater_equal.y:
            quadrants[1].append(particle)

        elif not greater_equal.x and not greater_equal.y:
            quadrants[2].append(particle)

        else:
            quadrants[3].append(particle)

    return quadrants


@dataclass(frozen=True)
class Quadtree:

    particles: Iterable[Vector2]
    center: Vector2
    side_length: float
    parent_node: Union["Quadtree, None"] = None
    child_nodes: Union["Quadtree", None] = field(init=False)

    def __post_init__(self):

        if len(self.particles) <= 1:
            object.__setattr__(self, "child_nodes", None)
            return

        child_side_length = self.side_length / 2
        child_offset = self.side_length / 4

        signs = map(Vector2, [(1, 1), (-1, 1), (-1, -1), (1, -1)])

        child_quadrants = subdivide(self.center, self.particles)

        child_nodes = [
            Quadtree(child_particles, self.center + child_offset * sign,
                     child_side_length, self)
            for child_particles, sign in zip(child_quadrants, signs)
        ]

        object.__setattr__(self, "child_nodes", child_nodes)

    @property
    def mass(self) -> float:
        return sum(particle.mass for particle in self.particles)
