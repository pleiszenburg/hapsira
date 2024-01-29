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
    "dense_output_hf",
]


A_EXTRA = _A[N_STAGES + 1 :]
C_EXTRA = _C[N_STAGES + 1 :]
D = _D


class DOP853:
    """
    Explicit Runge-Kutta method of order 8.
    """

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
            self.rr_old = self.rr
            self.vv_old = self.vv
            (
                self.h_previous,
                self.t,
                self.rr,
                self.vv,
                self.h_abs,
                self.fr,
                self.fv,
                self.K,
            ) = rets

        if not success:
            self.status = "failed"
            return

        self.t_old = t
        if self.direction * (self.t - self.t_bound) < 0:
            return

        self.status = "finished"


# TODO compile
def dense_output_hf(
    fun, argk, t_old, t, h_previous, rr, vv, rr_old, vv_old, fr, fv, K_
):
    """Compute a local interpolant over the last successful step.

    Returns
    -------
    sol : `DenseOutput`
        Local interpolant over the last successful step.
    """

    assert t_old is not None
    assert t != t_old

    Ke = np.empty((N_STAGES_EXTENDED, N_RV), dtype=float)
    Ke[: N_STAGES + 1, :] = np.array(K_)

    h = h_previous

    for s, (a, c) in enumerate(zip(A_EXTRA, C_EXTRA), start=N_STAGES + 1):
        dy = np.dot(Ke[:s].T, a[:s]) * h
        rr_ = add_VV_hf(rr_old, array_to_V_hf(dy[:3]))
        vv_ = add_VV_hf(vv_old, array_to_V_hf(dy[3:]))
        rr_, vv_ = fun(
            t_old + c * h,
            rr_,
            vv_,
            argk,
        )  # TODO call into hf
        Ke[s] = np.array([*rr_, *vv_])

    fr_old = array_to_V_hf(Ke[0, :3])
    fv_old = array_to_V_hf(Ke[0, 3:])

    delta_rr = sub_VV_hf(rr, rr_old)
    delta_vv = sub_VV_hf(vv, vv_old)

    F00 = *delta_rr, *delta_vv
    F01 = *sub_VV_hf(mul_Vs_hf(fr_old, h), delta_rr), *sub_VV_hf(
        mul_Vs_hf(fv_old, h), delta_vv
    )
    F02 = *sub_VV_hf(
        mul_Vs_hf(delta_rr, 2), mul_Vs_hf(add_VV_hf(fr, fr_old), h)
    ), *sub_VV_hf(mul_Vs_hf(delta_vv, 2), mul_Vs_hf(add_VV_hf(fv, fv_old), h))

    F03, F04, F05, F06 = tuple(
        tuple(float(number) for number in line) for line in (h * np.dot(D, Ke))
    )  # TODO

    return (
        t_old,
        t - t_old,  # h
        rr_old,
        vv_old,
        (F00, F01, F02, F03, F04, F05, F06),
    )
