import numpy as np
from dataclasses import dataclass
from vector import Vector


@dataclass
class Particle:
    position: Vector
    velocity: Vector
    mass: float
