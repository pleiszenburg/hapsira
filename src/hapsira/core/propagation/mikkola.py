from math import cos, cosh, log, sin, sinh, sqrt

from ..angles import (
    D_to_nu_hf,
    E_to_M_hf,
    E_to_nu_hf,
    F_to_M_hf,
    F_to_nu_hf,
    nu_to_E_hf,
    nu_to_F_hf,
)
from ..elements import coe2rv_hf, rv2coe_hf, RV2COE_TOL
from ..jit import array_to_V_hf, hjit, vjit, gjit


__all__ = [
    "mikkola_coe_hf",
    "mikkola_coe_vf",
    "mikkola_rv_hf",
    "mikkola_rv_gf",
]


@hjit("f(f,f,f,f,f,f,f,f)")
def mikkola_coe_hf(k, p, ecc, inc, raan, argp, nu, tof):
    """
    Scalar mikkola_coe
    """

    a = p / (1 - ecc**2)
    n = sqrt(k / abs(a) ** 3)

    # Solve for specific geometrical case
    if ecc < 1.0:
        # Equation (9a)
        alpha = (1 - ecc) / (4 * ecc + 1 / 2)
        M0 = E_to_M_hf(nu_to_E_hf(nu, ecc), ecc)
    else:
        alpha = (ecc - 1) / (4 * ecc + 1 / 2)
        M0 = F_to_M_hf(nu_to_F_hf(nu, ecc), ecc)

    M = M0 + n * tof
    beta = M / 2 / (4 * ecc + 1 / 2)

    # Equation (9b)
    if beta >= 0:
        z = (beta + sqrt(beta**2 + alpha**3)) ** (1 / 3)
    else:
        z = (beta - sqrt(beta**2 + alpha**3)) ** (1 / 3)

    s = z - alpha / z

    # Apply initial correction
    if ecc < 1.0:
        ds = -0.078 * s**5 / (1 + ecc)
    else:
        ds = 0.071 * s**5 / (1 + 0.45 * s**2) / (1 + 4 * s**2) / ecc

    s += ds

    # Solving for the true anomaly
    if ecc < 1.0:
        E = M + ecc * (3 * s - 4 * s**3)
        f = E - ecc * sin(E) - M
        f1 = 1.0 - ecc * cos(E)
        f2 = ecc * sin(E)
        f3 = ecc * cos(E)
        f4 = -f2
        f5 = -f3
    else:
        E = 3 * log(s + sqrt(1 + s**2))
        f = -E + ecc * sinh(E) - M
        f1 = -1.0 + ecc * cosh(E)
        f2 = ecc * sinh(E)
        f3 = ecc * cosh(E)
        f4 = f2
        f5 = f3

    # Apply Taylor expansion
    u1 = -f / f1
    u2 = -f / (f1 + 0.5 * f2 * u1)
    u3 = -f / (f1 + 0.5 * f2 * u2 + (1.0 / 6.0) * f3 * u2**2)
    u4 = -f / (
        f1 + 0.5 * f2 * u3 + (1.0 / 6.0) * f3 * u3**2 + (1.0 / 24.0) * f4 * (u3**3)
    )
    u5 = -f / (
        f1
        + f2 * u4 / 2
        + f3 * (u4 * u4) / 6.0
        + f4 * (u4 * u4 * u4) / 24.0
        + f5 * (u4 * u4 * u4 * u4) / 120.0
    )

    E += u5

    if ecc < 1.0:
        nu = E_to_nu_hf(E, ecc)
    else:
        if ecc == 1.0:
            # Parabolic
            nu = D_to_nu_hf(E)
        else:
            # Hyperbolic
            nu = F_to_nu_hf(E, ecc)

    return nu


@vjit("f(f,f,f,f,f,f,f,f)")
def mikkola_coe_vf(k, p, ecc, inc, raan, argp, nu, tof):
    """
    Vectorized mikkola_coe
    """

    return mikkola_coe_hf(k, p, ecc, inc, raan, argp, nu, tof)


@hjit("Tuple([V,V])(f,V,V,f)")
def mikkola_rv_hf(k, r0, v0, tof):
    """Raw algorithm for Mikkola's Kepler solver.

    Parameters
    ----------
    k : float
        Standard gravitational parameter of the attractor.
    r0 : tuple[float,float,float]
        Position vector.
    v0 : tuple[float,float,float]
        Velocity vector.
    tof : float
        Time of flight.

    Returns
    -------
    rr : tuple[float,float,float]
        Final velocity vector.
    vv : tuple[float,float,float]
        Final velocity vector.
    Note
    ----
    Original paper: https://doi.org/10.1007/BF01235850
    """
    # Solving for the classical elements
    p, ecc, inc, raan, argp, nu = rv2coe_hf(k, r0, v0, RV2COE_TOL)
    nu = mikkola_coe_hf(k, p, ecc, inc, raan, argp, nu, tof)

    return coe2rv_hf(k, p, ecc, inc, raan, argp, nu)


@gjit("void(f,f[:],f[:],f,f[:],f[:])", "(),(n),(n),()->(n),(n)")
def mikkola_rv_gf(k, r0, v0, tof, rr, vv):
    """
    Vectorized mikkola_rv
    """

    (rr[0], rr[1], rr[2]), (vv[0], vv[1], vv[2]) = mikkola_rv_hf(
        k, array_to_V_hf(r0), array_to_V_hf(v0), tof
    )
