from math import fabs, isnan

from ..linalg import EPS
from ...jit import hjit


__all__ = [
    "BRENTQ_CONVERGED",
    "BRENTQ_SIGNERR",
    "BRENTQ_CONVERR",
    "BRENTQ_ERROR",
    "BRENTQ_XTOL",
    "BRENTQ_RTOL",
    "BRENTQ_MAXITER",
    "brentq_hf",
]


BRENTQ_CONVERGED = 0
BRENTQ_SIGNERR = -1
BRENTQ_CONVERR = -2
BRENTQ_ERROR = -3

BRENTQ_XTOL = 2e-12
BRENTQ_RTOL = 4 * EPS
BRENTQ_MAXITER = 100


@hjit("f(f,f)")
def _min_ss_hf(a, b):
    return a if a < b else b


@hjit("b1(f)")
def _signbit_s_hf(a):
    return a < 0


@hjit("Tuple([f,i8])(F(f(f)),f,f,f,f,f)", forceobj=True, nopython=False, cache=False)
def brentq_hf(
    func,  # callback_type
    xa,  # double
    xb,  # double
    xtol,  # double
    rtol,  # double
    maxiter,  # int
):
    """
    Loosely adapted from
    https://github.com/scipy/scipy/blob/d23363809572e9a44074a3f06f66137083446b48/scipy/optimize/_zeros_py.py#L682
    """

    # if not xtol + 0. > 0:
    #     return 0., BRENTQ_ERROR
    # if not rtol + 0. >= BRENTQ_RTOL:
    #     return 0., BRENTQ_ERROR
    # if not maxiter + 0 >= 0:
    #     return 0., BRENTQ_ERROR

    xpre, xcur = xa, xb
    xblk = 0.0
    fpre, fcur, fblk = 0.0, 0.0, 0.0
    spre, scur = 0.0, 0.0

    fpre = func(xpre)
    if isnan(fpre):
        return 0.0, BRENTQ_ERROR

    fcur = func(xcur)
    if isnan(fcur):
        return 0.0, BRENTQ_ERROR

    if fpre == 0:
        return xpre, BRENTQ_CONVERGED
    if fcur == 0:
        return xcur, BRENTQ_CONVERGED
    if _signbit_s_hf(fpre) == _signbit_s_hf(fcur):
        return 0.0, BRENTQ_SIGNERR

    for _ in range(0, maxiter):
        if fpre != 0 and fcur != 0 and _signbit_s_hf(fpre) != _signbit_s_hf(fcur):
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
            return xcur, BRENTQ_CONVERGED

        if fabs(spre) > delta and fabs(fcur) < fabs(fpre):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = (
                    -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
                )
            if 2 * fabs(stry) < _min_ss_hf(fabs(spre), 3 * fabs(sbis) - delta):
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
        if isnan(fcur):
            return 0.0, BRENTQ_ERROR

    return xcur, BRENTQ_CONVERR
