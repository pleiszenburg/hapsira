from math import gamma, inf

from numba import njit as jit
import numpy as np

from ..jit import hjit, vjit

__all__ = [
    "hyp2f1b_hf",
    "hyp2f1b_vf",
    "stumpff_c2",
    "stumpff_c3",
]


@hjit("f(f)")
def hyp2f1b_hf(x):
    """Hypergeometric function 2F1(3, 1, 5/2, x), see [Battin].

    .. todo::
        Add more information about this function

    Notes
    -----
    More information about hypergeometric function can be checked at
    https://en.wikipedia.org/wiki/Hypergeometric_function

    """
    if x >= 1.0:
        return inf

    res = 1.0
    term = 1.0
    ii = 0
    while True:
        term = term * (3 + ii) * (1 + ii) / (5 / 2 + ii) * x / (ii + 1)
        res_old = res
        res += term
        if res_old == res:
            return res
        ii += 1


@vjit("f(f)")
def hyp2f1b_vf(x):
    """
    Vectorized hyp2f1b
    """

    return hyp2f1b_hf(x)


@jit
def stumpff_c2(psi):
    r"""Second Stumpff function.

    For positive arguments:

    .. math::

        c_2(\psi) = \frac{1 - \cos{\sqrt{\psi}}}{\psi}

    """
    eps = 1.0
    if psi > eps:
        res = (1 - np.cos(np.sqrt(psi))) / psi
    elif psi < -eps:
        res = (np.cosh(np.sqrt(-psi)) - 1) / (-psi)
    else:
        res = 1.0 / 2.0
        delta = (-psi) / gamma(2 + 2 + 1)
        k = 1
        while res + delta != res:
            res = res + delta
            k += 1
            delta = (-psi) ** k / gamma(2 * k + 2 + 1)

    return res


@jit
def stumpff_c3(psi):
    r"""Third Stumpff function.

    For positive arguments:

    .. math::

        c_3(\psi) = \frac{\sqrt{\psi} - \sin{\sqrt{\psi}}}{\sqrt{\psi^3}}

    """
    eps = 1.0
    if psi > eps:
        res = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / (psi * np.sqrt(psi))
    elif psi < -eps:
        res = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / (-psi * np.sqrt(-psi))
    else:
        res = 1.0 / 6.0
        delta = (-psi) / gamma(2 + 3 + 1)
        k = 1
        while res + delta != res:
            res = res + delta
            k += 1
            delta = (-psi) ** k / gamma(2 * k + 3 + 1)

    return res
