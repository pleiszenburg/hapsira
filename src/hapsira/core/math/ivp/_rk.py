from math import inf, sqrt
from typing import Callable

import numpy as np

from ._const import (
    N_RV,
    N_STAGES,
    KSIG,
    SAFETY,
    MIN_FACTOR,
    MAX_FACTOR,
    INTERPOLATOR_POWER,
    N_STAGES_EXTENDED,
    ERROR_ESTIMATOR_ORDER,
    ERROR_EXPONENT,
)
from ._dop853_coefficients import A as _A, C as _C, D as _D
from ._rkstep import rk_step_hf
from ._rkerror import estimate_error_norm_V_hf

from ...jit import array_to_V_hf, hjit, DSIG
from ...math.linalg import (
    abs_V_hf,
    add_Vs_hf,
    add_VV_hf,
    div_VV_hf,
    max_VV_hf,
    mul_Vs_hf,
    nextafter_hf,
    norm_VV_hf,
    sub_VV_hf,
    EPS,
)

__all__ = [
    "DOP853",
]


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
    d0 = norm_VV_hf(div_VV_hf(rr, scale_r), div_VV_hf(vv, scale_v)) * factor
    d1 = norm_VV_hf(div_VV_hf(fr, scale_r), div_VV_hf(fv, scale_v)) * factor

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
        norm_VV_hf(
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


@hjit(
    f"Tuple([b1,f,f,V,V,f,V,V,{KSIG:s}])"
    f"(F({DSIG:s}),f,f,V,V,V,V,f,f,f,f,f,{KSIG:s})"
)
def _step_impl_hf(
    fun, argk, t, rr, vv, fr, fv, rtol, atol, direction, h_abs, t_bound, K
):
    min_step = 10 * abs(nextafter_hf(t, direction * inf) - t)

    if h_abs < min_step:
        h_abs = min_step

    step_accepted = False
    step_rejected = False

    while not step_accepted:
        if h_abs < min_step:
            return (
                False,
                0.0,
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                K,
            )

        h = h_abs * direction
        t_new = t + h

        if direction * (t_new - t_bound) > 0:
            t_new = t_bound

        h = t_new - t
        h_abs = abs(h)

        rr_new, vv_new, fr_new, fv_new, K_new = rk_step_hf(
            fun,
            t,
            rr,
            vv,
            fr,
            fv,
            h,
            argk,
        )

        scale_r = add_Vs_hf(
            mul_Vs_hf(
                max_VV_hf(
                    abs_V_hf(rr),
                    abs_V_hf(rr_new),
                ),
                rtol,
            ),
            atol,
        )
        scale_v = add_Vs_hf(
            mul_Vs_hf(
                max_VV_hf(
                    abs_V_hf(vv),
                    abs_V_hf(vv_new),
                ),
                rtol,
            ),
            atol,
        )
        error_norm = estimate_error_norm_V_hf(
            K_new,
            h,
            scale_r,
            scale_v,
        )

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

    return True, h, t_new, rr_new, vv_new, h_abs, fr_new, fv_new, K_new


class Dop853DenseOutput:
    """local interpolant over step made by an ODE solver.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """

    def __init__(self, t_old, t, y_old, F):
        self.t_old = t_old
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

        y = y.reshape(6)
        return array_to_V_hf(y[:3]), array_to_V_hf(y[3:])


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
        rr: tuple,
        vv: tuple,
        t_bound: float,
        argk: float,
        rtol: float,
        atol: float,
    ):
        assert atol >= 0

        if rtol < 100 * EPS:
            rtol = 100 * EPS

        self.t = t0
        self.rr = rr
        self.vv = vv
        self.t_bound = t_bound
        self.fun = fun
        self.argk = argk
        self.rtol = rtol
        self.atol = atol

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1

        self.K_extended = np.empty(
            (N_STAGES_EXTENDED, N_RV), dtype=float
        )  # TODO set type
        self.K = self.K_extended[: N_STAGES + 1, :]
        self.rr_old = None
        self.vv_old = None
        self.t_old = None
        self.h_previous = None

        self.status = "running"

        self.fr, self.fv = self.fun(
            self.t,
            self.rr,
            self.vv,
            self.argk,
        )  # TODO call into hf

        self.h_abs = _select_initial_step_hf(
            self.fun,
            self.t,
            self.rr,
            self.vv,
            self.argk,
            self.fr,
            self.fv,
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
        success, *rets = _step_impl_hf(
            self.fun,
            self.argk,
            self.t,
            self.rr,
            self.vv,
            self.fr,
            self.fv,
            self.rtol,
            self.atol,
            self.direction,
            self.h_abs,
            self.t_bound,
            tuple(tuple(line) for line in self.K[: N_STAGES + 1, :N_RV]),
        )

        if success:
            self.h_previous = rets[0]
            # self.y_old = np.array([*rets[1], *rets[2]])
            self.rr_old = self.rr
            self.vv_old = self.vv
            self.t = rets[1]
            self.rr, self.vv = rets[2], rets[3]
            self.h_abs = rets[4]
            self.fr, self.fv = rets[5], rets[6]
            self.K[: N_STAGES + 1, :N_RV] = np.array(rets[7])

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
            rr_ = add_VV_hf(self.rr_old, array_to_V_hf(dy[:3]))
            vv_ = add_VV_hf(self.vv_old, array_to_V_hf(dy[3:]))
            rr, vv = self.fun(
                self.t_old + c * h,
                rr_,
                vv_,
                self.argk,
            )  # TODO call into hf
            K[s] = np.array([*rr, *vv])

        F = np.empty((INTERPOLATOR_POWER, N_RV), dtype=float)  # TODO use correct type

        fr_old = array_to_V_hf(K[0, :3])
        fv_old = array_to_V_hf(K[0, 3:])

        delta_rr = sub_VV_hf(self.rr, self.rr_old)
        delta_vv = sub_VV_hf(self.vv, self.vv_old)

        F[0, :3] = delta_rr
        F[0, 3:] = delta_vv

        F[1, :3] = sub_VV_hf(mul_Vs_hf(fr_old, h), delta_rr)
        F[1, 3:] = sub_VV_hf(mul_Vs_hf(fv_old, h), delta_vv)

        F[2, :3] = sub_VV_hf(
            mul_Vs_hf(delta_rr, 2), mul_Vs_hf(add_VV_hf(self.fr, fr_old), h)
        )
        F[2, 3:] = sub_VV_hf(
            mul_Vs_hf(delta_vv, 2), mul_Vs_hf(add_VV_hf(self.fv, fv_old), h)
        )

        F[3:, :] = h * np.dot(self.D, K)  # TODO

        return Dop853DenseOutput(
            self.t_old, self.t, np.array([*self.rr_old, *self.vv_old]), F
        )
