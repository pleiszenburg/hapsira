from math import (
    asinh,
    atan,
    atan2,
    atanh,
    cos,
    cosh,
    nan,
    pi,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)

from .jit import hjit, vjit


_TOL = 1.48e-08


@hjit("f(f,f)")
def E_to_M_hf(E, ecc):
    r"""Mean anomaly from eccentric anomaly.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    E : float
        Eccentric anomaly in radians.
    ecc : float
        Eccentricity.

    Returns
    -------
    M : float
        Mean anomaly.

    Warnings
    --------
    The mean anomaly will be outside of (-π, π]
    if the eccentric anomaly is.
    No validation or wrapping is performed.

    Notes
    -----
    The implementation uses the plain original Kepler equation:

    .. math::
        M = E - e \sin{E}

    """
    M = E - ecc * sin(E)
    return M


@vjit("f(f,f)")
def E_to_M_vf(E, ecc):
    """
    Vectorized E_to_M
    """

    return E_to_M_hf(E, ecc)


@hjit("f(f,f)")
def F_to_M_hf(F, ecc):
    r"""Mean anomaly from hyperbolic anomaly.

    Parameters
    ----------
    F : float
        Hyperbolic anomaly.
    ecc : float
        Eccentricity (>1).

    Returns
    -------
    M : float
        Mean anomaly.

    Notes
    -----
    As noted in [5]_, by manipulating
    the parametric equations of the hyperbola
    we can derive a quantity that is equivalent
    to the mean anomaly in the elliptic case:

    .. math::

        M = e \sinh{F} - F

    """
    M = ecc * sinh(F) - F
    return M


@vjit("f(f,f)")
def F_to_M_vf(F, ecc):
    """
    Vectorized F_to_M
    """

    return F_to_M_hf(F, ecc)


@hjit("f(f,f,f)")
def _kepler_equation_hf(E, M, ecc):
    return E_to_M_hf(E, ecc) - M


@hjit("f(f,f,f)")
def _kepler_equation_prime_hf(E, M, ecc):
    return 1 - ecc * cos(E)


@hjit("f(f,f,f)")
def _kepler_equation_hyper_hf(F, M, ecc):
    return F_to_M_hf(F, ecc) - M


@hjit("f(f,f,f)")
def _kepler_equation_prime_hyper_hf(F, M, ecc):
    return ecc * cosh(F) - 1


@hjit("f(f,f,f,f,i64)")
def _newton_elliptic_hf(p0, M, ecc, tol, maxiter):
    for _ in range(maxiter):
        fval = _kepler_equation_hf(p0, M, ecc)
        fder = _kepler_equation_prime_hf(p0, M, ecc)
        newton_step = fval / fder
        p = p0 - newton_step
        if abs(p - p0) < tol:
            return p
        p0 = p
    return nan


@hjit("f(f,f,f,f,i64)")
def _newton_hyperbolic_hf(p0, M, ecc, tol, maxiter):
    for _ in range(maxiter):
        fval = _kepler_equation_hyper_hf(p0, M, ecc)
        fder = _kepler_equation_prime_hyper_hf(p0, M, ecc)
        newton_step = fval / fder
        p = p0 - newton_step
        if abs(p - p0) < tol:
            return p
        p0 = p
    return nan


@hjit("f(f)")
def D_to_nu_hf(D):
    r"""True anomaly from parabolic anomaly.

    Parameters
    ----------
    D : float
        Eccentric anomaly.

    Returns
    -------
    nu : float
        True anomaly.

    Notes
    -----
    From [1]_:

    .. math::

        \nu = 2 \arctan{D}

    """
    return 2 * atan(D)


@vjit("f(f)")
def D_to_nu_vf(D):
    """
    Vectorized D_to_nu
    """

    return D_to_nu_hf(D)


@hjit("f(f)")
def nu_to_D_hf(nu):
    r"""Parabolic anomaly from true anomaly.

    Parameters
    ----------
    nu : float
        True anomaly in radians.

    Returns
    -------
    D : float
        Parabolic anomaly.

    Warnings
    --------
    The parabolic anomaly will be continuous in (-∞, ∞)
    only if the true anomaly is in (-π, π].
    No validation or wrapping is performed.

    Notes
    -----
    The treatment of the parabolic case is heterogeneous in the literature,
    and that includes the use of an equivalent quantity to the eccentric anomaly:
    [1]_ calls it "parabolic eccentric anomaly" D,
    [2]_ also uses the letter D but calls it just "parabolic anomaly",
    [3]_ uses the letter B citing indirectly [4]_
    (which however calls it "parabolic time argument"),
    and [5]_ does not bother to define it.

    We use this definition:

    .. math::

        B = \tan{\frac{\nu}{2}}

    References
    ----------
    .. [1] Farnocchia, Davide, Davide Bracali Cioci, and Andrea Milani.
       "Robust resolution of Kepler’s equation in all eccentricity regimes."
    .. [2] Bate, Muller, White.
    .. [3] Vallado, David. "Fundamentals of Astrodynamics and Applications",
       2013.
    .. [4] IAU VIth General Assembly, 1938.
    .. [5] Battin, Richard H. "An introduction to the Mathematics and Methods
       of Astrodynamics, Revised Edition", 1999.

    """
    # TODO: Rename to B
    return tan(nu / 2)


@vjit("f(f)")
def nu_to_D_vf(nu):
    """
    Vectorized nu_to_D
    """

    return nu_to_D_hf(nu)


@hjit("f(f,f)")
def nu_to_E_hf(nu, ecc):
    r"""Eccentric anomaly from true anomaly.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    nu : float
        True anomaly in radians.
    ecc : float
        Eccentricity.

    Returns
    -------
    E : float
        Eccentric anomaly, between -π and π radians.

    Warnings
    --------
    The eccentric anomaly will be between -π and π radians,
    no matter the value of the true anomaly.

    Notes
    -----
    The implementation uses the half-angle formula from [3]_:

    .. math::
        E = 2 \arctan \left ( \sqrt{\frac{1 - e}{1 + e}} \tan{\frac{\nu}{2}} \right)
        \in (-\pi, \pi]

    """
    E = 2 * atan(sqrt((1 - ecc) / (1 + ecc)) * tan(nu / 2))
    return E


@vjit("f(f,f)")
def nu_to_E_vf(nu, ecc):
    """
    Vectorized nu_to_E
    """

    return nu_to_E_hf(nu, ecc)


@hjit("f(f,f)")
def nu_to_F_hf(nu, ecc):
    r"""Hyperbolic anomaly from true anomaly.

    Parameters
    ----------
    nu : float
        True anomaly in radians.
    ecc : float
        Eccentricity (>1).

    Returns
    -------
    F : float
        Hyperbolic anomaly.

    Warnings
    --------
    The hyperbolic anomaly will be continuous in (-∞, ∞)
    only if the true anomaly is in (-π, π],
    which should happen anyway
    because the true anomaly is limited for hyperbolic orbits.
    No validation or wrapping is performed.

    Notes
    -----
    The implementation uses the half-angle formula from [3]_:

    .. math::
        F = 2 \operatorname{arctanh} \left( \sqrt{\frac{e-1}{e+1}} \tan{\frac{\nu}{2}} \right)

    """
    F = 2 * atanh(sqrt((ecc - 1) / (ecc + 1)) * tan(nu / 2))
    return F


@vjit("f(f,f)")
def nu_to_F_vf(nu, ecc):
    """
    Vectorized nu_to_F
    """

    return nu_to_F_hf(nu, ecc)


@hjit("f(f,f)")
def E_to_nu_hf(E, ecc):
    r"""True anomaly from eccentric anomaly.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    E : float
        Eccentric anomaly in radians.
    ecc : float
        Eccentricity.

    Returns
    -------
    nu : float
        True anomaly, between -π and π radians.

    Warnings
    --------
    The true anomaly will be between -π and π radians,
    no matter the value of the eccentric anomaly.

    Notes
    -----
    The implementation uses the half-angle formula from [3]_:

    .. math::
        \nu = 2 \arctan \left( \sqrt{\frac{1 + e}{1 - e}} \tan{\frac{E}{2}} \right)
        \in (-\pi, \pi]

    """
    nu = 2 * atan(sqrt((1 + ecc) / (1 - ecc)) * tan(E / 2))
    return nu


@vjit("f(f,f)")
def E_to_nu_vf(E, ecc):
    """
    Vectorized E_to_nu
    """

    return E_to_nu_hf(E, ecc)


@hjit("f(f,f)")
def F_to_nu_hf(F, ecc):
    r"""True anomaly from hyperbolic anomaly.

    Parameters
    ----------
    F : float
        Hyperbolic anomaly.
    ecc : float
        Eccentricity (>1).

    Returns
    -------
    nu : float
        True anomaly.

    Notes
    -----
    The implementation uses the half-angle formula from [3]_:

    .. math::
        \nu = 2 \arctan \left( \sqrt{\frac{e + 1}{e - 1}} \tanh{\frac{F}{2}} \right)
        \in (-\pi, \pi]

    """
    nu = 2 * atan(sqrt((ecc + 1) / (ecc - 1)) * tanh(F / 2))
    return nu


@vjit("f(f,f)")
def F_to_nu_vf(F, ecc):
    """
    Vectorized F_to_nu
    """

    return F_to_nu_hf(F, ecc)


@hjit("f(f,f)")
def M_to_E_hf(M, ecc):
    """Eccentric anomaly from mean anomaly.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    M : float
        Mean anomaly in radians.
    ecc : float
        Eccentricity.

    Returns
    -------
    E : float
        Eccentric anomaly.

    Notes
    -----
    This uses a Newton iteration on the Kepler equation.

    """
    if -pi < M < 0 or pi < M:
        E0 = M - ecc
    else:
        E0 = M + ecc
    E = _newton_elliptic_hf(E0, M, ecc, _TOL, 50)
    return E


@vjit("f(f,f)")
def M_to_E_vf(M, ecc):
    """
    Vectorized M_to_E
    """

    return M_to_E_hf(M, ecc)


@hjit("f(f,f)")
def M_to_F_hf(M, ecc):
    """Hyperbolic anomaly from mean anomaly.

    Parameters
    ----------
    M : float
        Mean anomaly in radians.
    ecc : float
        Eccentricity (>1).

    Returns
    -------
    F : float
        Hyperbolic anomaly.

    Notes
    -----
    This uses a Newton iteration on the hyperbolic Kepler equation.

    """
    F0 = asinh(M / ecc)
    F = _newton_hyperbolic_hf(F0, M, ecc, _TOL, 100)
    return F


@vjit("f(f,f)")
def M_to_F_vf(M, ecc):
    """
    Vectorized M_to_F
    """

    return M_to_F_hf(M, ecc)


@hjit("f(f)")
def M_to_D_hf(M):
    """Parabolic anomaly from mean anomaly.

    Parameters
    ----------
    M : float
        Mean anomaly in radians.

    Returns
    -------
    D : float
        Parabolic anomaly.

    Notes
    -----
    This uses the analytical solution of Barker's equation from [5]_.

    """
    B = 3.0 * M / 2.0
    A = (B + (1.0 + B**2) ** 0.5) ** (2.0 / 3.0)
    D = 2 * A * B / (1 + A + A**2)
    return D


@vjit("f(f)")
def M_to_D_vf(M):
    """
    Vectorized M_to_D
    """

    return M_to_D_hf(M)


@hjit("f(f)")
def D_to_M_hf(D):
    r"""Mean anomaly from parabolic anomaly.

    Parameters
    ----------
    D : float
        Parabolic anomaly.

    Returns
    -------
    M : float
        Mean anomaly.

    Notes
    -----
    We use this definition:

    .. math::

        M = B + \frac{B^3}{3}

    Notice that M < ν until ν ~ 100 degrees,
    then it reaches π when ν ~ 120 degrees,
    and grows without bounds after that.
    Therefore, it can hardly be called an "anomaly"
    since it is by no means an angle.

    """
    M = D + D**3 / 3
    return M


@vjit("f(f)")
def D_to_M_vf(D):
    """
    Vectorized D_to_M
    """

    return D_to_M_hf(D)


@hjit("f(f,f)")
def fp_angle_hf(nu, ecc):
    r"""Returns the flight path angle.

    Parameters
    ----------
    nu : float
        True anomaly in radians.
    ecc : float
        Eccentricity.

    Returns
    -------
    fp_angle: float
        Flight path angle

    Notes
    -----
    From [3]_, pp. 113:

    .. math::

        \phi = \arctan(\frac {e \sin{\nu}}{1 + e \cos{\nu}})

    """
    return atan2(ecc * sin(nu), 1 + ecc * cos(nu))


@vjit("f(f,f)")
def fp_angle_vf(nu, ecc):
    """
    Vectorized fp_angle
    """

    return fp_angle_hf(nu, ecc)
