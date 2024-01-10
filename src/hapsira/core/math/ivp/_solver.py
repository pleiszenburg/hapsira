from numba import njit, jit

from math import fabs


CONVERGED = 0
SIGNERR = -1
CONVERR = -2
# EVALUEERR = -3
INPROGRESS = 1


@njit
def _min_hf(a, b):
    return a if a < b else b


@njit
def _signbit_hf(a):
    return a < 0


@jit
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


@jit
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
