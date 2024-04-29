from math import acos, acosh, cos, cosh, nan, pi, sqrt

from ..angles import (
    D_to_M_hf,
    D_to_nu_hf,
    E_to_M_hf,
    E_to_nu_hf,
    F_to_M_hf,
    F_to_nu_hf,
    M_to_D_hf,
    M_to_E_hf,
    M_to_F_hf,
    nu_to_D_hf,
    nu_to_E_hf,
    nu_to_F_hf,
)
from ..elements import coe2rv_hf, rv2coe_hf, RV2COE_TOL
from ..jit import array_to_V_hf, hjit, vjit, gjit


__all__ = [
    "delta_t_from_nu_hf",
    "delta_t_from_nu_vf",
    "farnocchia_coe_hf",
    "farnocchia_coe_vf",
    "farnocchia_rv_hf",
    "farnocchia_rv_gf",
    "FARNOCCHIA_K",
    "FARNOCCHIA_Q",
    "FARNOCCHIA_DELTA",
]


FARNOCCHIA_K = 1.0
FARNOCCHIA_Q = 1.0
FARNOCCHIA_DELTA = 1e-2

_ATOL = 1e-12
_TOL = 1.48e-08
_MAXITER = 50


@hjit("f(f,f,f)")
def _S_x_hf(ecc, x, atol):
    assert abs(x) < 1
    S = 0
    k = 0
    while True:
        S_old = S
        S += (ecc - 1 / (2 * k + 3)) * x**k
        k += 1
        if abs(S - S_old) < atol:
            return S


@hjit("f(f,f,f)")
def _dS_x_alt_hf(ecc, x, atol):
    # Notice that this is not exactly
    # the partial derivative of S with respect to D,
    # but the result of arranging the terms
    # in section 4.2 of Farnocchia et al. 2013
    assert abs(x) < 1
    S = 0
    k = 0
    while True:
        S_old = S
        S += (ecc - 1 / (2 * k + 3)) * (2 * k + 3) * x**k
        k += 1
        if abs(S - S_old) < atol:
            return S


@hjit("f(f,f,f)")
def _d2S_x_alt_hf(ecc, x, atol):
    # Notice that this is not exactly
    # the second partial derivative of S with respect to D,
    # but the result of arranging the terms
    # in section 4.2 of Farnocchia et al. 2013
    # Also, notice that we are not using this function yet
    assert abs(x) < 1
    S = 0
    k = 0
    while True:
        S_old = S
        S += (ecc - 1 / (2 * k + 3)) * (2 * k + 3) * (2 * k + 2) * x**k
        k += 1
        if abs(S - S_old) < atol:
            return S


@hjit("f(f,f,f)")
def _kepler_equation_prime_near_parabolic_hf(D, M, ecc):
    x = (ecc - 1.0) / (ecc + 1.0) * (D**2)
    assert abs(x) < 1
    S = _dS_x_alt_hf(ecc, x, _ATOL)
    return sqrt(2.0 / (1.0 + ecc)) + sqrt(2.0 / (1.0 + ecc) ** 3) * (D**2) * S


@hjit("f(f,f)")
def _D_to_M_near_parabolic_hf(D, ecc):
    x = (ecc - 1.0) / (ecc + 1.0) * (D**2)
    assert abs(x) < 1
    S = _S_x_hf(ecc, x, _ATOL)
    return sqrt(2.0 / (1.0 + ecc)) * D + sqrt(2.0 / (1.0 + ecc) ** 3) * (D**3) * S


@hjit("f(f,f,f)")
def _kepler_equation_near_parabolic_hf(D, M, ecc):
    return _D_to_M_near_parabolic_hf(D, ecc) - M


@hjit("f(f,f,f,i8)")
def _M_to_D_near_parabolic_hf(M, ecc, tol, maxiter):
    """Parabolic eccentric anomaly from mean anomaly, near parabolic case.

    Parameters
    ----------
    M : float
        Mean anomaly in radians.
    ecc : float
        Eccentricity (~1).
    tol : float, optional
        Absolute tolerance for Newton convergence.
    maxiter : int, optional
        Maximum number of iterations for Newton convergence.

    Returns
    -------
    D : float
        Parabolic eccentric anomaly.

    """
    D0 = M_to_D_hf(M)

    for _ in range(maxiter):
        fval = _kepler_equation_near_parabolic_hf(D0, M, ecc)
        fder = _kepler_equation_prime_near_parabolic_hf(D0, M, ecc)

        newton_step = fval / fder
        D = D0 - newton_step
        if abs(D - D0) < tol:
            return D

        D0 = D

    return nan


@hjit("f(f,f,f,f,f)")
def delta_t_from_nu_hf(nu, ecc, k, q, delta):
    """Time elapsed since periapsis for given true anomaly.

    Parameters
    ----------
    nu : float
        True anomaly.
    ecc : float
        Eccentricity.
    k : float
        Gravitational parameter.
    q : float
        Periapsis distance.
    delta : float
        Parameter that controls the size of the near parabolic region.

    Returns
    -------
    delta_t : float
        Time elapsed since periapsis.

    """
    assert -pi <= nu < pi
    if ecc < 1 - delta:
        # Strong elliptic
        E = nu_to_E_hf(nu, ecc)  # (-pi, pi]
        M = E_to_M_hf(E, ecc)  # (-pi, pi]
        n = sqrt(k * (1 - ecc) ** 3 / q**3)
    elif 1 - delta <= ecc < 1:
        E = nu_to_E_hf(nu, ecc)  # (-pi, pi]
        if delta <= 1 - ecc * cos(E):
            # Strong elliptic
            M = E_to_M_hf(E, ecc)  # (-pi, pi]
            n = sqrt(k * (1 - ecc) ** 3 / q**3)
        else:
            # Near parabolic
            D = nu_to_D_hf(nu)  # (-∞, ∞)
            # If |nu| is far from pi this result is bounded
            # because the near parabolic region shrinks in its vicinity,
            # otherwise the eccentricity is very close to 1
            # and we are really far away
            M = _D_to_M_near_parabolic_hf(D, ecc)
            n = sqrt(k / (2 * q**3))
    elif ecc == 1:
        # Parabolic
        D = nu_to_D_hf(nu)  # (-∞, ∞)
        M = D_to_M_hf(D)  # (-∞, ∞)
        n = sqrt(k / (2 * q**3))
    elif 1 + ecc * cos(nu) < 0:
        # Unfeasible region
        return nan
    elif 1 < ecc <= 1 + delta:
        # NOTE: Do we need to wrap nu here?
        # For hyperbolic orbits, it should anyway be in
        # (-arccos(-1 / ecc), +arccos(-1 / ecc))
        F = nu_to_F_hf(nu, ecc)  # (-∞, ∞)
        if delta <= ecc * cosh(F) - 1:
            # Strong hyperbolic
            M = F_to_M_hf(F, ecc)  # (-∞, ∞)
            n = sqrt(k * (ecc - 1) ** 3 / q**3)
        else:
            # Near parabolic
            D = nu_to_D_hf(nu)  # (-∞, ∞)
            M = _D_to_M_near_parabolic_hf(D, ecc)  # (-∞, ∞)
            n = sqrt(k / (2 * q**3))
    elif 1 + delta < ecc:
        # Strong hyperbolic
        F = nu_to_F_hf(nu, ecc)  # (-∞, ∞)
        M = F_to_M_hf(F, ecc)  # (-∞, ∞)
        n = sqrt(k * (ecc - 1) ** 3 / q**3)
    else:
        raise RuntimeError

    return M / n


@vjit("f(f,f,f,f,f)")
def delta_t_from_nu_vf(nu, ecc, k, q, delta):
    """
    Vectorized delta_t_from_nu
    """

    return delta_t_from_nu_hf(nu, ecc, k, q, delta)


@hjit("f(f,f,f,f,f)")
def _nu_from_delta_t_hf(delta_t, ecc, k, q, delta):
    """True anomaly for given elapsed time since periapsis.

    Parameters
    ----------
    delta_t : float
        Time elapsed since periapsis.
    ecc : float
        Eccentricity.
    k : float
        Gravitational parameter.
    q : float
        Periapsis distance.
    delta : float
        Parameter that controls the size of the near parabolic region.

    Returns
    -------
    nu : float
        True anomaly.

    """
    if ecc < 1 - delta:
        # Strong elliptic
        n = sqrt(k * (1 - ecc) ** 3 / q**3)
        M = n * delta_t
        # This might represent several revolutions,
        # so we wrap the true anomaly
        E = M_to_E_hf((M + pi) % (2 * pi) - pi, ecc)
        nu = E_to_nu_hf(E, ecc)
    elif 1 - delta <= ecc < 1:
        E_delta = acos((1 - delta) / ecc)
        # We compute M assuming we are in the strong elliptic case
        # and verify later
        n = sqrt(k * (1 - ecc) ** 3 / q**3)
        M = n * delta_t
        # We check against abs(M) because E_delta could also be negative
        if E_to_M_hf(E_delta, ecc) <= abs(M):
            # Strong elliptic, proceed
            # This might represent several revolutions,
            # so we wrap the true anomaly
            E = M_to_E_hf((M + pi) % (2 * pi) - pi, ecc)
            nu = E_to_nu_hf(E, ecc)
        else:
            # Near parabolic, recompute M
            n = sqrt(k / (2 * q**3))
            M = n * delta_t
            D = _M_to_D_near_parabolic_hf(M, ecc, _TOL, _MAXITER)
            nu = D_to_nu_hf(D)
    elif ecc == 1:
        # Parabolic
        n = sqrt(k / (2 * q**3))
        M = n * delta_t
        D = M_to_D_hf(M)
        nu = D_to_nu_hf(D)
    elif 1 < ecc <= 1 + delta:
        F_delta = acosh((1 + delta) / ecc)
        # We compute M assuming we are in the strong hyperbolic case
        # and verify later
        n = sqrt(k * (ecc - 1) ** 3 / q**3)
        M = n * delta_t
        # We check against abs(M) because F_delta could also be negative
        if F_to_M_hf(F_delta, ecc) <= abs(M):
            # Strong hyperbolic, proceed
            F = M_to_F_hf(M, ecc)
            nu = F_to_nu_hf(F, ecc)
        else:
            # Near parabolic, recompute M
            n = sqrt(k / (2 * q**3))
            M = n * delta_t
            D = _M_to_D_near_parabolic_hf(M, ecc, _TOL, _MAXITER)
            nu = D_to_nu_hf(D)
    # elif 1 + delta < ecc:
    else:
        # Strong hyperbolic
        n = sqrt(k * (ecc - 1) ** 3 / q**3)
        M = n * delta_t
        F = M_to_F_hf(M, ecc)
        nu = F_to_nu_hf(F, ecc)

    return nu


@hjit("f(f,f,f,f,f,f,f,f)")
def farnocchia_coe_hf(k, p, ecc, inc, raan, argp, nu, tof):
    """
    Scalar farnocchia_coe
    """

    q = p / (1.0 + ecc)

    delta_t0 = delta_t_from_nu_hf(nu, ecc, k, q, FARNOCCHIA_DELTA)
    delta_t = delta_t0 + tof

    return _nu_from_delta_t_hf(delta_t, ecc, k, q, FARNOCCHIA_DELTA)


@vjit("f(f,f,f,f,f,f,f,f)")
def farnocchia_coe_vf(k, p, ecc, inc, raan, argp, nu, tof):
    """
    Vectorized farnocchia_coe
    """

    return farnocchia_coe_hf(k, p, ecc, inc, raan, argp, nu, tof)


@hjit("Tuple([V,V])(f,V,V,f)")
def farnocchia_rv_hf(k, r0, v0, tof):
    r"""Propagates orbit using mean motion.

    This algorithm depends on the geometric shape of the orbit.
    For the case of the strong elliptic or strong hyperbolic orbits:

    ..  math::

        M = M_{0} + \frac{\mu^{2}}{h^{3}}\left ( 1 -e^{2}\right )^{\frac{3}{2}}t

    .. versionadded:: 0.9.0

    Parameters
    ----------
    k : float
        Standar Gravitational parameter
    r0 : tuple[float,float,float]
        Initial position vector wrt attractor center.
    v0 : tuple[float,float,float]
        Initial velocity vector.
    tof : float
        Time of flight (s).

    Notes
    -----
    This method takes initial :math:`\vec{r}, \vec{v}`, calculates classical orbit parameters,
    increases mean anomaly and performs inverse transformation to get final :math:`\vec{r}, \vec{v}`
    The logic is based on formulae (4), (6) and (7) from http://dx.doi.org/10.1007/s10569-013-9476-9

    """
    # get the initial true anomaly and orbit parameters that are constant over time
    p, ecc, inc, raan, argp, nu0 = rv2coe_hf(k, r0, v0, RV2COE_TOL)
    nu = farnocchia_coe_hf(k, p, ecc, inc, raan, argp, nu0, tof)

    return coe2rv_hf(k, p, ecc, inc, raan, argp, nu)


@gjit(
    "void(f,f[:],f[:],f,f[:],f[:])",
    "(),(n),(n),()->(n),(n)",
)
def farnocchia_rv_gf(k, r0, v0, tof, rr, vv):
    """
    Vectorized farnocchia_rv
    """

    (rr[0], rr[1], rr[2]), (vv[0], vv[1], vv[2]) = farnocchia_rv_hf(
        k, array_to_V_hf(r0), array_to_V_hf(v0), tof
    )
