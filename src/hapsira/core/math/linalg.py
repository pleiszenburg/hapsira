from numba import njit as jit
import numpy as np

__all__ = [
    "norm",
]


@jit
def norm(arr):
    return np.sqrt(arr @ arr)
