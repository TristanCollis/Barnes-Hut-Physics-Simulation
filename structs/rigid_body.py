from typing import Protocol

from structs.vector import Vector


class RigidBody(Protocol):
    mass: float
    position: Vector