import numpy as np

from scipy.interpolate import interp1d as _scipy_interp1d

from numpy import asarray, array
from numpy import searchsorted

__all__ = [
    "interp1d",
    "spline_interp",
    "sinc_interp",
]


def _check_broadcast_up_to(arr_from, shape_to, name):
    """Helper to check that arr_from broadcasts up to shape_to"""
    shape_from = arr_from.shape
    if len(shape_to) >= len(shape_from):
        for t, f in zip(shape_to[::-1], shape_from[::-1]):
            if f != 1 and f != t:
                break
        else:  # all checks pass, do the upcasting that we need later
            if arr_from.size != 1 and arr_from.shape != shape_to:
                arr_from = np.ones(shape_to, arr_from.dtype) * arr_from
            return arr_from.ravel()
    # at least one check failed
    raise ValueError(
        f"{name} argument must be able to broadcast up "
        f"to shape {shape_to} but had shape {shape_from}"
    )


class interp1d:
    def __init__(
        self,
        x,
        y,
    ):
        self._y_axis = -1
        self._y_extra_shape = None
        self._set_yi(y, xi=x, axis=-1)

        x = array(x, copy=True)
        y = array(y, copy=True)

        ind = np.argsort(x, kind="mergesort")
        x = x[ind]
        y = np.take(y, ind, axis=-1)

        assert x.ndim == 1
        assert y.ndim != 0

        # Backward compatibility
        self.axis = -1 % y.ndim

        # Interpolation goes internally along the first axis
        self.y = y
        self._y = self._reshape_yi(self.y)
        self.x = x

        assert len(self.x) >= 1

        broadcast_shape = self.y.shape[: self.axis] + self.y.shape[self.axis + 1 :]
        if len(broadcast_shape) == 0:
            broadcast_shape = (1,)
        # it's either a pair (_below_range, _above_range) or a single value
        # for both above and below range
        fill_value = np.asarray(np.nan)
        below_above = [
            _check_broadcast_up_to(fill_value, broadcast_shape, "fill_value")
        ] * 2
        self._fill_value_below, self._fill_value_above = below_above
        # backwards compat: fill_value was a public attr; make it writeable
        self._fill_value_orig = fill_value

    def __call__(self, x):
        x, x_shape = self._prepare_x(x)
        y = self._evaluate(x)
        return self._finish_y(y, x_shape)

    def _prepare_x(self, x):
        """Reshape input x array to 1-D"""
        x = array(x)
        x_shape = x.shape
        return x.ravel(), x_shape

    def _reshape_yi(self, yi, check=False):
        yi = np.moveaxis(np.asarray(yi), self._y_axis, 0)
        if check and yi.shape[1:] != self._y_extra_shape:
            ok_shape = "{!r} + (N,) + {!r}".format(
                self._y_extra_shape[-self._y_axis :],
                self._y_extra_shape[: -self._y_axis],
            )
            raise ValueError("Data must be of shape %s" % ok_shape)
        return yi.reshape((yi.shape[0], -1))

    def _finish_y(self, y, x_shape):
        """Reshape interpolated y back to an N-D array similar to initial y"""
        y = y.reshape(x_shape + self._y_extra_shape)
        if self._y_axis != 0 and x_shape != ():
            nx = len(x_shape)
            ny = len(self._y_extra_shape)
            s = (
                list(range(nx, nx + self._y_axis))
                + list(range(nx))
                + list(range(nx + self._y_axis, nx + ny))
            )
            y = y.transpose(s)
        return y

    def _set_yi(self, yi, xi=None, axis=None):
        if axis is None:
            axis = self._y_axis
        if axis is None:
            raise ValueError("no interpolation axis specified")

        yi = np.asarray(yi)

        shape = yi.shape
        if shape == ():
            shape = (1,)
        if xi is not None and shape[axis] != len(xi):
            raise ValueError(
                "x and y arrays must be equal in length along " "interpolation axis."
            )

        self._y_axis = axis % yi.ndim
        self._y_extra_shape = yi.shape[: self._y_axis] + yi.shape[self._y_axis + 1 :]

    def _evaluate(self, x_new):
        x_new = asarray(x_new)

        # 2. Find where in the original data, the values to interpolate
        #    would be inserted.
        #    Note: If x_new[n] == x[m], then m is returned by searchsorted.
        x_new_indices = searchsorted(self.x, x_new)

        # 3. Clip x_new_indices so that they are within the range of
        #    self.x indices and at least 1. Removes mis-interpolation
        #    of x_new[n] = x[0]
        x_new_indices = x_new_indices.clip(1, len(self.x) - 1).astype(int)

        # 4. Calculate the slope of regions that each x_new value falls in.
        lo = x_new_indices - 1
        hi = x_new_indices

        x_lo = self.x[lo]
        x_hi = self.x[hi]
        y_lo = self._y[lo]
        y_hi = self._y[hi]

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
