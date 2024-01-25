from math import fabs, isnan

from ..linalg import EPS
from ...jit import hjit


CONVERGED = 0
SIGNERR = -1
CONVERR = -2

BRENTQ_ITER = 100
BRENTQ_XTOL = 2e-12
BRENTQ_RTOL = 4 * EPS


@hjit("f(f,f)")
def _min_ss_hf(a, b):
    return a if a < b else b


@hjit("b1(f)")
def _signbit_s_hf(a):
    return a < 0


@hjit("f(F(f(f)),f,f,f,f,f)", forceobj=True, nopython=False, cache=False)
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

    fpre = func(xpre)
    assert not isnan(fpre)
    fcur = func(xcur)
    assert not isnan(fcur)
    if fpre == 0:
        return xpre, CONVERGED
    if fcur == 0:
        return xcur, CONVERGED
    if _signbit_s_hf(fpre) == _signbit_s_hf(fcur):
        return 0.0, SIGNERR

    for _ in range(0, iter_):
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
            return xcur, CONVERGED

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
        assert not isnan(fcur)

    return xcur, CONVERR


@hjit("f(F(f(f)),f,f,f,f,f)", forceobj=True, nopython=False, cache=False)
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

    assert xtol > 0
    assert rtol >= BRENTQ_RTOL
    assert maxiter >= 0

    zero, error_num = _brentq_hf(
        f,
        a,
        b,
        xtol,
        rtol,
        maxiter,
    )

    assert error_num == CONVERGED

    return zero
