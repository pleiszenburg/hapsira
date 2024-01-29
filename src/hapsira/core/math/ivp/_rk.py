from typing import Callable

import numpy as np

from ._const import ERROR_ESTIMATOR_ORDER
from ._rkstepinit import select_initial_step_hf
from ._rkstepimpl import step_impl_hf
from ..ieee754 import EPS

__all__ = [
    "DOP853",
]


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
