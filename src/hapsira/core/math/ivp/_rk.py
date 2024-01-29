from typing import Callable

import numpy as np

from ._const import (
    N_RV,
    N_STAGES,
    N_STAGES_EXTENDED,
    ERROR_ESTIMATOR_ORDER,
)
from ._dop853_coefficients import A as _A, C as _C, D as _D
from ._rkstepinit import select_initial_step_hf
from ._rkstepimpl import step_impl_hf


from ...jit import array_to_V_hf
from ...math.linalg import (
    add_VV_hf,
    mul_Vs_hf,
    sub_VV_hf,
    EPS,
)

__all__ = [
    "DOP853",
]


class Dop853DenseOutput:
    """local interpolant over step made by an ODE solver.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """

    def __init__(self, t_old, h, rr_old, vv_old, F):
        self.t_old = t_old
        self.h = h
        self._F = F
        self.rr_old = rr_old
        self.vv_old = vv_old

    def __call__(self, t: float):
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

        F00, F01, F02, F03, F04, F05, F06 = self._F

        x = (t - self.t_old) / self.h
        rr_new = (0.0, 0.0, 0.0)
        vv_new = (0.0, 0.0, 0.0)

        for idx, f in enumerate((F06, F05, F04, F03, F02, F01, F00)):
            rr_new = add_VV_hf(rr_new, f[:3])
            vv_new = add_VV_hf(vv_new, f[3:])

            if idx % 2 == 0:
                rr_new = mul_Vs_hf(rr_new, x)
                vv_new = mul_Vs_hf(vv_new, x)
            else:
                rr_new = mul_Vs_hf(rr_new, 1 - x)
                vv_new = mul_Vs_hf(vv_new, 1 - x)

        rr_new = add_VV_hf(rr_new, self.rr_old)
        vv_new = add_VV_hf(vv_new, self.vv_old)

        return rr_new, vv_new


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

        self.K = (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 0
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 1
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 2
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 3
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 4
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 5
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 6
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 7
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 8
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 9
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 10
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 11
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 12
        )

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

        self.h_abs = select_initial_step_hf(
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
        success, *rets = step_impl_hf(
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
            self.K,
        )

        if success:
            self.h_previous = rets[0]
            self.rr_old = self.rr
            self.vv_old = self.vv
            self.t = rets[1]
            self.rr, self.vv = rets[2], rets[3]
            self.h_abs = rets[4]
            self.fr, self.fv = rets[5], rets[6]
            self.K = rets[7]

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

        K = np.empty((N_STAGES_EXTENDED, N_RV), dtype=float)
        K[: N_STAGES + 1, :] = np.array(self.K)

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

        fr_old = array_to_V_hf(K[0, :3])
        fv_old = array_to_V_hf(K[0, 3:])

        delta_rr = sub_VV_hf(self.rr, self.rr_old)
        delta_vv = sub_VV_hf(self.vv, self.vv_old)

        F00 = *delta_rr, *delta_vv
        F01 = *sub_VV_hf(mul_Vs_hf(fr_old, h), delta_rr), *sub_VV_hf(
            mul_Vs_hf(fv_old, h), delta_vv
        )
        F02 = *sub_VV_hf(
            mul_Vs_hf(delta_rr, 2), mul_Vs_hf(add_VV_hf(self.fr, fr_old), h)
        ), *sub_VV_hf(mul_Vs_hf(delta_vv, 2), mul_Vs_hf(add_VV_hf(self.fv, fv_old), h))

        F03, F04, F05, F06 = tuple(
            tuple(float(number) for number in line) for line in (h * np.dot(self.D, K))
        )  # TODO

        return Dop853DenseOutput(
            self.t_old,
            self.t - self.t_old,  # h
            self.rr_old,
            self.vv_old,
            (F00, F01, F02, F03, F04, F05, F06),
        )
