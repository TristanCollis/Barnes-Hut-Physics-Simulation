import numpy as np
import numpy.typing as npt
from typing import Annotated, Literal, TypeVar, Union, Unpack, Any

Vector = np.ndarray[float, Any]

norm = np.linalg.norm

def vec2(x: float, y: float) -> Vector:
    try:
        return np.array([x, y], dtype=float)
    except:
        raise TypeError("x and y must be of type 'float'")