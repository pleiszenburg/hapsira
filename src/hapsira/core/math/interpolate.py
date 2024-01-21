import numpy as np
from numba import njit

from scipy.interpolate import interp1d as _scipy_interp1d

__all__ = [
    "interp1d",
    "spline_interp",
    "sinc_interp",
]


@njit
def bisect_left(a, x):
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


class interp1d:
    def __init__(
        self,
        x,
        y,
    ):
        assert x.ndim == 1
        assert y.ndim == 2
        assert x.shape[0] >= 1  # > instead of >=
        assert y.shape[0] == 3
        assert y.shape[1] == x.shape[0]

        self.y = y.T
        self.x = np.array(x, copy=True)

        self._fill_value_below = np.array([np.nan])
        self._fill_value_above = np.array([np.nan])

    def __call__(self, x):
        "x is scalar"

        x_new = np.array([x])

        # 2. Find where in the original data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
        x_new_indices = np.array(
            [bisect_left(self.x, x_new[0])]
        )  # np.searchsorted(self.x, x_new)

        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1. Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = x_new_indices.clip(1, len(self.x) - 1).astype(int)

        # 4. Calculate the slope of regions that each x_new value falls in.
        lo = x_new_indices - 1
        hi = x_new_indices

        x_lo = self.x[lo]
        x_hi = self.x[hi]
        y_lo = self.y[lo]
        y_hi = self.y[hi]

        # Note that the following two expressions rely on the specifics of the
        # broadcasting semantics.
        slope = (y_hi - y_lo) / (x_hi - x_lo)[:, None]

        # 5. Calculate the actual value for each entry in x_new.
        y_new = slope * (x_new - x_lo)[:, None] + y_lo

        below_bounds, above_bounds = self._check_bounds(x_new)
        if len(y_new) > 0:
            # Note fill_value must be broadcast up to the proper size
            # and flattened to work here
            y_new[below_bounds] = self._fill_value_below
            y_new[above_bounds] = self._fill_value_above

        y_new = y_new.reshape((3,))
        return y_new

    def _check_bounds(self, x_new):
        """Check the inputs for being in the bounds of the interpolated data.

        Parameters
        ----------
        x_new : array

        Returns
        -------
        out_of_bounds : bool array
            The mask on x_new of values that are out of the bounds.
        """

        # If self.bounds_error is True, we raise an error if any x_new values
        # fall outside the range of x. Otherwise, we return an array indicating
        # which values are outside the boundary region.
        below_bounds = x_new < self.x[0]
        above_bounds = x_new > self.x[-1]

        if below_bounds.any():
            below_bounds_value = x_new[np.argmax(below_bounds)]
            raise ValueError(
                f"A value ({below_bounds_value}) in x_new is below "
                f"the interpolation range's minimum value ({self.x[0]})."
            )
        if above_bounds.any():
            above_bounds_value = x_new[np.argmax(above_bounds)]
            raise ValueError(
                f"A value ({above_bounds_value}) in x_new is above "
                f"the interpolation range's maximum value ({self.x[-1]})."
            )

        # !! Should we emit a warning if some values are out of bounds?
        # !! matlab does not.
        return below_bounds, above_bounds


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
