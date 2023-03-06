from typing import Protocol

import numpy as np

from structs import Particle, Quadtree, RigidBody, Vector, norm, vec2

G = 1

def gravity(point1: Vector, point2: Vector, mass1: float, mass2: float) -> Vector:
    r = point2 - point1
    if norm(r) == 0:
        return vec2(0, 0)
    
    return G * mass1 * mass2 * r / norm(r)**3