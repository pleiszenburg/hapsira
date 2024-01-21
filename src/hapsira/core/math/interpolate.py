from typing import Callable

import numpy as np
from scipy.interpolate import interp1d as _scipy_interp1d

from ..jit import hjit
from .linalg import add_VV_hf, div_Vs_hf, mul_Vs_hf, sub_VV_hf

__all__ = [
    "interp_hb",
    "spline_interp",
    "sinc_interp",
]


def interp_hb(x: np.ndarray, y: np.ndarray) -> Callable:
    """
    Build compiled 1d-interpolator for 1D vectors
    """

    assert x.ndim == 1
    assert y.ndim == 2
    assert x.shape[0] >= 1  # > instead of >=
    assert y.shape[0] == 3
    assert y.shape[1] == x.shape[0]

    y = tuple(tuple(record) for record in y.T)
    x = tuple(x)
    x_len = len(x)

    @hjit("V(f)")
    def interp_hf(x_new):
        assert x_new >= x[0]
        assert x_new <= x[-1]

        # bisect left
        x_new_index = 0
        hi = x_len
        while x_new_index < hi:
            mid = (x_new_index + hi) // 2
            if x[mid] < x_new:
                x_new_index = mid + 1
            else:
                hi = mid

        # clip
        if x_new_index > x_len:
            x_new_index = x_len
        if x_new_index < 1:
            x_new_index = 1

        # slope
        lo = x_new_index - 1
        hi = x_new_index
        x_lo = x[lo]
        x_hi = x[hi]
        y_lo = y[lo]  # tuple
        y_hi = y[hi]  # tuple
        slope = div_Vs_hf(sub_VV_hf(y_hi, y_lo), x_hi - x_lo)  # tuple

        # new value
        y_new = add_VV_hf(mul_Vs_hf(slope, x_new - x_lo), y_lo)  # tuple

        return y_new

    return interp_hf


def spline_interp(y, x, u, *, kind="cubic"):
    """Interpolates y, sampled at x instants, at u instants using `scipy.interpolate.interp1d`."""
    y_u = _scipy_interp1d(x, y, kind=kind)(u)
    return y_u


def sinc_interp(y, x, u):
    """Interpolates y, sampled at x instants, at u instants using sinc interpolation.

    Notes
    -----
    Taken from https://gist.github.com/endolith/1297227.
    Possibly equivalent to `scipy.signal.resample`,
    see https://mail.python.org/pipermail/scipy-user/2012-January/031255.html.
    However, quick experiments show different ringing behavior.

    """
    if len(y) != len(x):
        raise ValueError("x and s must be the same length")

    # Find the period and assume it's constant
    T = x[1] - x[0]

    sincM = np.tile(u, (len(x), 1)) - np.tile(x[:, np.newaxis], (1, len(u)))
    y_u = y @ np.sinc(sincM / T)

    return y_u
