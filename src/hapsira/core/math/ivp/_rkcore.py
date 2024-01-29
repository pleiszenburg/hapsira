from math import nan
from typing import Callable

from ._const import ERROR_ESTIMATOR_ORDER
from ._rkstepinit import select_initial_step_hf
from ._rkstepimpl import step_impl_hf
from ..ieee754 import EPS
from ..linalg import sign_hf

__all__ = [
    "dop853_init_hf",
    "dop853_step_hf",
    "DOP853_RUNNING",
    "DOP853_FINISHED",
    "DOP853_FAILED",
    "DOP853_ARGK",
    "DOP853_FR",
    "DOP853_FUN",
    "DOP853_FV",
    "DOP853_H_PREVIOUS",
    "DOP853_K",
    "DOP853_RR",
    "DOP853_RR_OLD",
    "DOP853_STATUS",
    "DOP853_T",
    "DOP853_T_OLD",
    "DOP853_VV",
    "DOP853_VV_OLD",
]


DOP853_RUNNING = 0
DOP853_FINISHED = 1
DOP853_FAILED = 2

DOP853_ARGK = 5
DOP853_FR = 15
DOP853_FUN = 4
DOP853_FV = 16
DOP853_H_PREVIOUS = 13
DOP853_K = 9
DOP853_RR = 1
DOP853_RR_OLD = 10
DOP853_STATUS = 14
DOP853_T = 0
DOP853_T_OLD = 12
DOP853_VV = 2
DOP853_VV_OLD = 11


# TODO compile
def dop853_init_hf(
    fun: Callable,
    t0: float,
    rr: tuple,
    vv: tuple,
    t_bound: float,
    argk: float,
    rtol: float,
    atol: float,
):
    """
    Explicit Runge-Kutta method of order 8.
    """

    assert atol >= 0

    if rtol < 100 * EPS:
        rtol = 100 * EPS

    direction = sign_hf(t_bound - t0) if t_bound != t0 else 1

    K = (
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

    rr_old = (nan, nan, nan)
    vv_old = (nan, nan, nan)
    t_old = nan
    h_previous = nan

    status = DOP853_RUNNING

    fr, fv = fun(
        t0,
        rr,
        vv,
        argk,
    )

    h_abs = select_initial_step_hf(
        fun,
        t0,
        rr,
        vv,
        argk,
        fr,
        fv,
        direction,
        ERROR_ESTIMATOR_ORDER,
        rtol,
        atol,
    )

    return (
        t0,  # 0 -> t
        rr,  # 1
        vv,  # 2
        t_bound,  # 3
        fun,  # 4
        argk,  # 5
        rtol,  # 6
        atol,  # 7
        direction,  # 8
        K,  # 9
        rr_old,  # 10
        vv_old,  # 11
        t_old,  # 12
        h_previous,  # 13
        status,  # 14
        fr,  # 15
        fv,  # 16
        h_abs,  # 17
    )


# TODO compile
def dop853_step_hf(
    t,
    rr,
    vv,
    t_bound,
    fun,
    argk,
    rtol,
    atol,
    direction,
    K,
    rr_old,
    vv_old,
    t_old,
    h_previous,
    status,
    fr,
    fv,
    h_abs,
):
    """Perform one integration step.

    Returns
    -------
    message : string or None
        Report from the solver. Typically a reason for a failure if
        `self.status` is 'failed' after the step was taken or None
        otherwise.
    """

    if status != DOP853_RUNNING:
        raise RuntimeError("Attempt to step on a failed or finished " "solver.")

    if t == t_bound:
        # Handle corner cases of empty solver or no integration.
        t_old = t
        t = t_bound
        status = DOP853_FINISHED
        return (
            t,
            rr,
            vv,
            t_bound,
            fun,
            argk,
            rtol,
            atol,
            direction,
            K,
            rr_old,
            vv_old,
            t_old,
            h_previous,
            status,
            fr,
            fv,
            h_abs,
        )

    t_tmp = t
    success, *rets = step_impl_hf(
        fun,
        argk,
        t,
        rr,
        vv,
        fr,
        fv,
        rtol,
        atol,
        direction,
        h_abs,
        t_bound,
        K,
    )

    if success:
        rr_old = rr
        vv_old = vv
        (
            h_previous,
            t,
            rr,
            vv,
            h_abs,
            fr,
            fv,
            K,
        ) = rets

    if not success:
        status = DOP853_FAILED
    else:
        t_old = t_tmp
        if not direction * (t - t_bound) < 0:
            status = DOP853_FINISHED

    return (
        t,
        rr,
        vv,
        t_bound,
        fun,
        argk,
        rtol,
        atol,
        direction,
        K,
        rr_old,
        vv_old,
        t_old,
        h_previous,
        status,
        fr,
        fv,
        h_abs,
    )
