import numpy as np
from typing import Iterable


class Vector(np.ndarray):

    def __new__(cls, input_array: Iterable[float]) -> "Self":
        obj = np.asarray(input_array).view(cls)

        return obj


class Vector2(Vector):

    def __new__(cls, input_array: Iterable[float]):
        obj = np.asarray(input_array).view(cls)

        if len(obj) != 2:
            raise ValueError("vector2 must contain exactly 2 elements")

        return obj

    @property
    def x(self) -> float:
        return self[0]

    @x.setter
    def x(self, new_x) -> None:
        self[0] = new_x

    @property
    def y(self) -> float:
        return self[1]

    @y.setter
    def y(self, new_y) -> None:
        self[1] = new_y

    @property
    def norm_squared(self) -> float:
        return np.sum(self * self)

    @property
    def norm(self) -> float:
        return self.norm_squared**0.5


class Vector3(Vector2):

    def __new__(cls, input_array: Iterable[float]):
        obj = np.asarray(input_array).view(cls)

        if len(obj) != 3:
            raise ValueError("vector3 must contain exactly 3 elements")

        return obj

    @property
    def z(self) -> float:
        return self[2]

    @z.setter
    def z(self, new_z) -> None:
        self[0] = new_z


def vec(*components: Iterable[float]) -> Vector:
    return Vector(components)


def vec2(x: float, y: float) -> Vector2:
    return Vector2((x, y))


def vec3(x: float, y: float, z: float) -> Vector3:
    return Vector3((x, y, z))
