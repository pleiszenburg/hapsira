from math import sqrt
from typing import Callable

import numpy as np

from ._dop853_coefficients import A as _A, C as _C, D as _D
from ._rkstep import rk_step_hf, N_RV, N_STAGES
from ._rkerror import estimate_error_norm_hf

from ...jit import array_to_V_hf, hjit, DSIG
from ...math.linalg import add_VV_hf, div_VV_hf, mul_Vs_hf, sub_VV_hf

__all__ = [
    "EPS",
    "DOP853",
]


EPS = np.finfo(float).eps

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

INTERPOLATOR_POWER = 7
N_STAGES_EXTENDED = 16
ERROR_ESTIMATOR_ORDER = 7
ERROR_EXPONENT = -1 / (ERROR_ESTIMATOR_ORDER + 1)


@hjit("f(V,V)")
def _norm_VV_hf(x, y):
    return sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + y[0] ** 2 + y[1] ** 2 + y[2] ** 2)


@hjit(f"f(F({DSIG:s}),f,V,V,f,V,V,f,f,f,f)")
def _select_initial_step_hf(
    fun, t0, rr, vv, argk, fr, fv, direction, order, rtol, atol
):
    scale_r = (
        atol + abs(rr[0]) * rtol,
        atol + abs(rr[1]) * rtol,
        atol + abs(rr[2]) * rtol,
    )
    scale_v = (
        atol + abs(vv[0]) * rtol,
        atol + abs(vv[1]) * rtol,
        atol + abs(vv[2]) * rtol,
    )

    factor = 1 / sqrt(6)
    d0 = _norm_VV_hf(div_VV_hf(rr, scale_r), div_VV_hf(vv, scale_v)) * factor
    d1 = _norm_VV_hf(div_VV_hf(fr, scale_r), div_VV_hf(fv, scale_v)) * factor

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    yr1 = add_VV_hf(rr, mul_Vs_hf(fr, h0 * direction))
    yv1 = add_VV_hf(vv, mul_Vs_hf(fv, h0 * direction))

    fr1, fv1 = fun(
        t0 + h0 * direction,
        yr1,
        yv1,
        argk,
    )

    d2 = (
        _norm_VV_hf(
            div_VV_hf(sub_VV_hf(fr1, fr), scale_r),
            div_VV_hf(sub_VV_hf(fv1, fv), scale_v),
        )
        / h0
    )

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1)


class Dop853DenseOutput:
    """local interpolant over step made by an ODE solver.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """

    def __init__(self, t_old, t, y_old, F):
        self.t_old = t_old
        self.t = t
        self.t_min = min(t, t_old)
        self.t_max = max(t, t_old)
        self.h = t - t_old
        self.F = F
        self.y_old = y_old

    def __call__(self, t):
        """Evaluate the interpolant.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate the solution at.

        Returns
        -------
        y : ndarray, shape (n,) or (n, n_points)
            Computed values. Shape depends on whether `t` was a scalar or a
            1-D array.
        """
        t = np.asarray(t)
        assert not t.ndim > 1

        x = (t - self.t_old) / self.h

        if t.ndim == 0:
            y = np.zeros_like(self.y_old)
        else:
            x = x[:, None]
            y = np.zeros((len(x), len(self.y_old)), dtype=self.y_old.dtype)

        for i, f in enumerate(reversed(self.F)):
            y += f
            if i % 2 == 0:
                y *= x
            else:
                y *= 1 - x
        y += self.y_old

        return y.T


class DOP853:
    """
    Explicit Runge-Kutta method of order 8.
    """

    A_EXTRA = _A[N_STAGES + 1 :]
    C_EXTRA = _C[N_STAGES + 1 :]
    D = _D

    def __init__(
        self,
        fun: Callable,
        t0: float,
        y0: np.array,
        t_bound: float,
        argk: float,
        max_step: float = np.inf,
        rtol: float = 1e-3,
        atol: float = 1e-6,
    ):
        assert y0.shape == (N_RV,)
        assert np.isfinite(y0).all()
        assert max_step > 0
        assert atol >= 0

        if rtol < 100 * EPS:
            rtol = 100 * EPS

        self.t = t0
        self.y = y0
        self.t_bound = t_bound
        self.max_step = max_step
        self.fun = fun
        self.argk = argk
        self.rtol = rtol
        self.atol = atol

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1

        self.K_extended = np.empty((N_STAGES_EXTENDED, N_RV), dtype=self.y.dtype)
        self.K = self.K_extended[: N_STAGES + 1, :]
        self.y_old = None
        self.t_old = None
        self.h_previous = None

        self.status = "running"

        rr, vv = self.fun(
            self.t,
            array_to_V_hf(self.y[:3]),
            array_to_V_hf(self.y[3:]),
            self.argk,
        )  # TODO call into hf
        self.f = np.array([*rr, *vv])

        self.h_abs = _select_initial_step_hf(
            self.fun,
            self.t,
            array_to_V_hf(self.y[:3]),
            array_to_V_hf(self.y[3:]),
            self.argk,
            array_to_V_hf(self.f[:3]),
            array_to_V_hf(self.f[3:]),
            self.direction,
            ERROR_ESTIMATOR_ORDER,
            self.rtol,
            self.atol,
        )  # TODO call into hf

    def step(self):
        """Perform one integration step.

        Returns
        -------
        message : string or None
            Report from the solver. Typically a reason for a failure if
            `self.status` is 'failed' after the step was taken or None
            otherwise.
        """
        if self.status != "running":
            raise RuntimeError("Attempt to step on a failed or finished " "solver.")

        if self.t == self.t_bound:
            # Handle corner cases of empty solver or no integration.
            self.t_old = self.t
            self.t = self.t_bound
            self.status = "finished"
            return

        t = self.t
        success = self._step_impl()

        if not success:
            self.status = "failed"
            return

        self.t_old = t
        if self.direction * (self.t - self.t_bound) < 0:
            return

        self.status = "finished"

    def dense_output(self):
        """Compute a local interpolant over the last successful step.

        Returns
        -------
        sol : `DenseOutput`
            Local interpolant over the last successful step.
        """
        assert self.t_old is not None

        assert self.t != self.t_old

        K = self.K_extended
        h = self.h_previous
        for s, (a, c) in enumerate(zip(self.A_EXTRA, self.C_EXTRA), start=N_STAGES + 1):
            dy = np.dot(K[:s].T, a[:s]) * h
            y_ = self.y_old + dy
            rr, vv = self.fun(
                self.t_old + c * h,
                array_to_V_hf(y_[:3]),
                array_to_V_hf(y_[3:]),
                self.argk,
            )  # TODO call into hf
            K[s] = np.array([*rr, *vv])

        F = np.empty((INTERPOLATOR_POWER, N_RV), dtype=self.y_old.dtype)

        f_old = K[0]
        delta_y = self.y - self.y_old

        F[0] = delta_y
        F[1] = h * f_old - delta_y
        F[2] = 2 * delta_y - h * (self.f + f_old)
        F[3:] = h * np.dot(self.D, K)

        return Dop853DenseOutput(self.t_old, self.t, self.y_old, F)

    def _step_impl(self):
        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h_abs < min_step:
                return False

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            rr_new, vv_new, fr_new, fv_new, K_new = rk_step_hf(
                self.fun,
                t,
                array_to_V_hf(y[:3]),
                array_to_V_hf(y[3:]),
                array_to_V_hf(self.f[:3]),
                array_to_V_hf(self.f[3:]),
                h,
                self.argk,
            )  # TODO call into hf
            y_new = np.array([*rr_new, *vv_new])
            f_new = np.array([*fr_new, *fv_new])
            self.K[: N_STAGES + 1, :N_RV] = np.array([K_new])

            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            assert scale.shape == (N_RV,)
            error_norm = estimate_error_norm_hf(
                K_new,
                h,
                array_to_V_hf(scale[:3]),
                array_to_V_hf(scale[3:]),
            )  # TODO call into hf

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * error_norm**ERROR_EXPONENT)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR, SAFETY * error_norm**ERROR_EXPONENT)
                step_rejected = True

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True
