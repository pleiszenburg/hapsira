from math import cos, cosh, gamma, inf, sin, sinh, sqrt

from ..jit import hjit, vjit

__all__ = [
    "hyp2f1b_hf",
    "hyp2f1b_vf",
    "stumpff_c2_hf",
    "stumpff_c2_vf",
    "stumpff_c3_hf",
    "stumpff_c3_vf",
]


@hjit("f(f)")
def hyp2f1b_hf(x):
    """
    Hypergeometric function 2F1(3, 1, 5/2, x), see [Battin].

    .. todo::
        Add more information about this function

    Notes
    -----
    More information about hypergeometric function can be checked at
    https://en.wikipedia.org/wiki/Hypergeometric_function

    Parameters
    ----------
    x : float
        Scalar

    Returns
    -------
    res : float
        Scalar

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
    Hypergeometric function 2F1(3, 1, 5/2, x), see [Battin].

    .. todo::
        Add more information about this function

    Notes
    -----
    More information about hypergeometric function can be checked at
    https://en.wikipedia.org/wiki/Hypergeometric_function

    Parameters
    ----------
    x : float
        Scalar

    Returns
    -------
    b : float
        Scalar

    """

    return hyp2f1b_hf(x)


@hjit("f(f)")
def stumpff_c2_hf(psi):
    r"""
    Second Stumpff function.

    For positive arguments:

    .. math::

        c_2(\psi) = \frac{1 - \cos{\sqrt{\psi}}}{\psi}

    Parameters
    ----------
    psi : float
        Scalar

    Returns
    -------
    res : float
        Scalar

    """

    eps = 1.0

    if psi > eps:
        return (1 - cos(sqrt(psi))) / psi

    if psi < -eps:
        return (cosh(sqrt(-psi)) - 1) / (-psi)

    res = 1.0 / 2.0
    delta = (-psi) / gamma(2 + 2 + 1)
    k = 1
    while res + delta != res:
        res = res + delta
        k += 1
        delta = (-psi) ** k / gamma(2 * k + 2 + 1)
    return res


@vjit("f(f)")
def stumpff_c2_vf(psi):
    r"""
    Second Stumpff function.

    For positive arguments:

    .. math::

        c_2(\psi) = \frac{1 - \cos{\sqrt{\psi}}}{\psi}

    Parameters
    ----------
    psi : float
        Scalar

    Returns
    -------
    res : float
        Scalar

    """

    return stumpff_c2_hf(psi)


@hjit("f(f)")
def stumpff_c3_hf(psi):
    r"""
    Third Stumpff function.

    For positive arguments:

    .. math::

        c_3(\psi) = \frac{\sqrt{\psi} - \sin{\sqrt{\psi}}}{\sqrt{\psi^3}}

    Parameters
    ----------
    psi : float
        Scalar

    Returns
    -------
    res : float
        Scalar

    """

    eps = 1.0

    if psi > eps:
        return (sqrt(psi) - sin(sqrt(psi))) / (psi * sqrt(psi))

    if psi < -eps:
        return (sinh(sqrt(-psi)) - sqrt(-psi)) / (-psi * sqrt(-psi))

    res = 1.0 / 6.0
    delta = (-psi) / gamma(2 + 3 + 1)
    k = 1
    while res + delta != res:
        res = res + delta
        k += 1
        delta = (-psi) ** k / gamma(2 * k + 3 + 1)
    return res


@vjit("f(f)")
def stumpff_c3_vf(psi):
    r"""
    Third Stumpff function.

    For positive arguments:

    .. math::

        c_3(\psi) = \frac{\sqrt{\psi} - \sin{\sqrt{\psi}}}{\sqrt{\psi^3}}

    Parameters
    ----------
    psi : float
        Scalar

    Returns
    -------
    res : float
        Scalar

    """

    return stumpff_c3_hf(psi)
