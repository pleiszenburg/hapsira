from math import nan

from ._const import ERROR_ESTIMATOR_ORDER, KSIG
from ._rkstepinit import select_initial_step_hf
from ._rkstepimpl import step_impl_hf
from ..ieee754 import EPS
from ..linalg import sign_hf
from ...jit import hjit, DSIG

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

DOP853_SIG = f"f,V,V,f,F({DSIG}),f,f,f,f,{KSIG:s},V,V,f,f,f,V,V,f"


@hjit(f"Tuple([{DOP853_SIG:s}])(F({DSIG}),f,V,V,f,f,f,f)")
def dop853_init_hf(fun, t0, rr, vv, t_bound, argk, rtol, atol):
    """
    Explicit Runge-Kutta method of order 8.
    Functional re-write of constructor of class `DOP853` within `scipy.integrate`.

    Based on
    - https://github.com/scipy/scipy/blob/4edfcaa3ce8a387450b6efce968572def71be089/scipy/integrate/_ivp/rk.py#L502
    - https://github.com/scipy/scipy/blob/4edfcaa3ce8a387450b6efce968572def71be089/scipy/integrate/_ivp/rk.py#L85
    - https://github.com/scipy/scipy/blob/4edfcaa3ce8a387450b6efce968572def71be089/scipy/integrate/_ivp/base.py#L131

    Parameters
    ----------
    fun : float
        Right-hand side of the system.
    t0 : float
        Initial time.
    rr : float
        Initial state 0:3
    vv : float
        Initial state 3:6
    t_bound : float
        Boundary time
    argk : float
        Standard gravitational parameter for `fun`
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance

    Returns
    -------
    t0 : float
        Initial time.
    rr : tuple[float,float,float]
        Initial state 0:3
    vv : tuple[float,float,float]
        Initial state 3:6
    t_bound : float
        Boundary time
    fun : Callable
        Right-hand side of the system
    argk : float
        Standard gravitational parameter for `fun`
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    direction : float
        Integration direction
    K : tuple[[float,...],...]
        Storage array for RK stages
    rr_old : tuple[float,float,float]
        Last state 0:3
    vv_old : tuple[float,float,float]
        Last state 3:6
    t_old : float
        Last time
    h_previous : float
        Last step length
    status : float
        Solver status
    fr : tuple[float,float,float]
        Current value of the derivative 0:3
    fv : tuple[float,float,float]
        Current value of the derivative 3:6
    h_abs : float
        Absolute step

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


@hjit(f"Tuple([{DOP853_SIG:s}])({DOP853_SIG:s})")
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
    """
    Perform one integration step.
    Functional re-write of method `step` of class `OdeSolver` within `scipy.integrate`.

    Based on
    https://github.com/scipy/scipy/blob/4edfcaa3ce8a387450b6efce968572def71be089/scipy/integrate/_ivp/base.py#L175

    Parameters
    ----------
    t : float
        Current time.
    rr : tuple[float,float,float]
        Current state 0:3
    vv : tuple[float,float,float]
        Current state 3:6
    t_bound : float
        Boundary time
    fun : Callable
        Right-hand side of the system
    argk : float
        Standard gravitational parameter for `fun`
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    direction : float
        Integration direction
    K : tuple[[float,...],...]
        Storage array for RK stages
    rr_old : tuple[float,float,float]
        Last state 0:3
    vv_old : tuple[float,float,float]
        Last state 3:6
    t_old : float
        Last time
    h_previous : float
        Last step length
    status : float
        Solver status
    fr : tuple[float,float,float]
        Current value of the derivative 0:3
    fv : tuple[float,float,float]
        Current value of the derivative 3:6
    h_abs : float
        Absolute step

    Returns
    -------
    t : float
        Current time.
    rr : tuple[float,float,float]
        Current state 0:3
    vv : tuple[float,float,float]
        Current state 3:6
    t_bound : float
        Boundary time
    fun : Callable
        Right-hand side of the system
    argk : float
        Standard gravitational parameter for `fun`
    rtol : float
        Relative tolerance
    atol : float
        Absolute tolerance
    direction : float
        Integration direction
    K : tuple[[float,...],...]
        Storage array for RK stages
    rr_old : tuple[float,float,float]
        Last state 0:3
    vv_old : tuple[float,float,float]
        Last state 3:6
    t_old : float
        Last time
    h_previous : float
        Last step length
    status : float
        Solver status
    fr : tuple[float,float,float]
        Current value of the derivative 0:3
    fv : tuple[float,float,float]
        Current value of the derivative 3:6
    h_abs : float
        Absolute step

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
    rets = step_impl_hf(
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
    success = rets[0]

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
        ) = rets[1:]

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
