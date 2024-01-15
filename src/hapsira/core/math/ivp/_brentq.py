from math import fabs

import operator

from numba import njit, jit
import numpy as np


CONVERGED = 0
SIGNERR = -1
CONVERR = -2
# EVALUEERR = -3
INPROGRESS = 1

BRENTQ_ITER = 100
BRENTQ_XTOL = 2e-12
BRENTQ_RTOL = 4 * np.finfo(float).eps


@njit
def _min_hf(a, b):
    return a if a < b else b


@njit
def _signbit_hf(a):
    return a < 0


@jit(nopython=False)
def _brentq_hf(
    func,  # callback_type
    xa,  # double
    xb,  # double
    xtol,  # double
    rtol,  # double
    iter_,  # int
):
    xpre, xcur = xa, xb
    xblk = 0.0
    fpre, fcur, fblk = 0.0, 0.0, 0.0
    spre, scur = 0.0, 0.0

    iterations = 0

    fpre = func(xpre)
    fcur = func(xcur)
    funcalls = 2
    if fpre == 0:
        return xpre, funcalls, iterations, CONVERGED
    if fcur == 0:
        return xcur, funcalls, iterations, CONVERGED
    if _signbit_hf(fpre) == _signbit_hf(fcur):
        return 0.0, funcalls, iterations, SIGNERR

    iterations = 0
    for _ in range(0, iter_):
        iterations += 1
        if fpre != 0 and fcur != 0 and _signbit_hf(fpre) != _signbit_hf(fcur):
            xblk = xpre
            fblk = fpre
            scur = xcur - xpre
            spre = scur
        if fabs(fblk) < fabs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol * fabs(xcur)) / 2
        sbis = (xblk - xcur) / 2
        if fcur == 0 or fabs(sbis) < delta:
            return xcur, funcalls, iterations, CONVERGED

        if fabs(spre) > delta and fabs(fcur) < fabs(fpre):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = (
                    -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
                )
            if 2 * fabs(stry) < _min_hf(fabs(spre), 3 * fabs(sbis) - delta):
                spre = scur
                scur = stry
            else:
                spre = sbis
                scur = sbis
        else:
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if fabs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = func(xcur)
        funcalls += 1

    return xcur, funcalls, iterations, CONVERR


@jit(nopython=False)
def brentq_sf(
    func,  # func
    a,  # double
    b,  # double
    xtol,  # double
    rtol,  # double
    iter_,  # int
):
    if xtol < 0:
        raise ValueError("xtol must be >= 0")
    if iter_ < 0:
        raise ValueError("maxiter should be > 0")

    zero, funcalls, iterations, error_num = _brentq_hf(
        func,
        a,
        b,
        xtol,
        rtol,
        iter_,
    )

    if error_num == SIGNERR:
        raise ValueError("f(a) and f(b) must have different signs")
    if error_num == CONVERR:
        raise RuntimeError("Failed to converge after %d iterations." % iterations)

    return zero  # double


def _wrap_nan_raise(f):
    def f_raise(x):
        fx = f(x)
        f_raise._function_calls += 1
        if np.isnan(fx):
            msg = f"The function value at x={x} is NaN; " "solver cannot continue."
            err = ValueError(msg)
            err._x = x
            err._function_calls = f_raise._function_calls
            raise err
        return fx

    f_raise._function_calls = 0
    return f_raise


def brentq(
    f,
    a,
    b,
    xtol=BRENTQ_XTOL,
    rtol=BRENTQ_RTOL,
    maxiter=BRENTQ_ITER,
):
    """
    Loosely adapted from
    https://github.com/scipy/scipy/blob/d23363809572e9a44074a3f06f66137083446b48/scipy/optimize/_zeros_py.py#L682
    """
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    if rtol < BRENTQ_RTOL:
        raise ValueError(f"rtol too small ({rtol:g} < {BRENTQ_RTOL:g})")
    f = _wrap_nan_raise(f)
    r = brentq_sf(f, a, b, xtol, rtol, maxiter)
    return r
