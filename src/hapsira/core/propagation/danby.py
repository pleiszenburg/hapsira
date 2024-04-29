from math import atan2, cos, cosh, floor, log, pi, sin, sinh, sqrt

from ..angles import E_to_M_hf, F_to_M_hf, nu_to_E_hf, nu_to_F_hf
from ..elements import coe2rv_hf, rv2coe_hf, RV2COE_TOL
from ..math.linalg import sign_hf
from ..jit import array_to_V_hf, hjit, gjit, vjit


__all__ = [
    "danby_coe_hf",
    "danby_coe_vf",
    "danby_rv_hf",
    "danby_rv_gf",
    "DANBY_NUMITER",
    "DANBY_RTOL",
]


DANBY_NUMITER = 20
DANBY_RTOL = 1e-8


@hjit("f(f,f,f,f,f,f,f,f,i8,f)")
def danby_coe_hf(k, p, ecc, inc, raan, argp, nu, tof, numiter, rtol):
    """
    Scalar danby_coe
    """

    semi_axis_a = p / (1 - ecc**2)
    n = sqrt(k / abs(semi_axis_a) ** 3)

    if ecc == 0:
        # Solving for circular orbit
        M0 = nu  # for circular orbit M = E = nu
        M = M0 + n * tof
        nu = M - 2 * pi * floor(M / 2 / pi)
        return nu

    elif ecc < 1.0:
        # For elliptical orbit
        M0 = E_to_M_hf(nu_to_E_hf(nu, ecc), ecc)
        M = M0 + n * tof
        xma = M - 2 * pi * floor(M / 2 / pi)
        E = xma + 0.85 * sign_hf(sin(xma)) * ecc

    else:
        # For parabolic and hyperbolic
        M0 = F_to_M_hf(nu_to_F_hf(nu, ecc), ecc)
        M = M0 + n * tof
        xma = M - 2 * pi * floor(M / 2 / pi)
        E = log(2 * xma / ecc + 1.8)

    # Iterations begin
    n = 0
    while n <= numiter:
        if ecc < 1.0:
            s = ecc * sin(E)
            c = ecc * cos(E)
            f = E - s - xma
            fp = 1 - c
            fpp = s
            fppp = c
        else:
            s = ecc * sinh(E)
            c = ecc * cosh(E)
            f = s - E - xma
            fp = c - 1
            fpp = s
            fppp = c

        if abs(f) <= rtol:
            if ecc < 1.0:
                sta = sqrt(1 - ecc**2) * sin(E)
                cta = cos(E) - ecc
            else:
                sta = sqrt(ecc**2 - 1) * sinh(E)
                cta = ecc - cosh(E)

            nu = atan2(sta, cta)
            break
        else:
            delta = -f / fp
            delta_star = -f / (fp + 0.5 * delta * fpp)
            deltak = -f / (fp + 0.5 * delta_star * fpp + delta_star**2 * fppp / 6)
            E = E + deltak
            n += 1
    else:
        raise ValueError("Maximum number of iterations has been reached.")

    return nu


@vjit("f(f,f,f,f,f,f,f,f,i8,f)")
def danby_coe_vf(k, p, ecc, inc, raan, argp, nu, tof, numiter, rtol):
    """
    Vectorized danby_coe
    """

    return danby_coe_hf(k, p, ecc, inc, raan, argp, nu, tof, numiter, rtol)


@hjit("Tuple([V,V])(f,V,V,f,i8,f)")
def danby_rv_hf(k, r0, v0, tof, numiter, rtol):
    """Kepler solver for both elliptic and parabolic orbits based on Danby's
    algorithm.

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
    numiter : int, optional
        Number of iterations, defaults to 20.
    rtol : float, optional
        Relative error for accuracy of the method, defaults to 1e-8.

    Returns
    -------
    rr : tuple[float,float,float]
        Final position vector.
    vv : tuple[float,float,float]
        Final velocity vector.

    Notes
    -----
    This algorithm was developed by Danby in his paper *The solution of Kepler
    Equation* with DOI: https://doi.org/10.1007/BF01686811
    """
    # Solve first for eccentricity and mean anomaly
    p, ecc, inc, raan, argp, nu = rv2coe_hf(k, r0, v0, RV2COE_TOL)
    nu = danby_coe_hf(k, p, ecc, inc, raan, argp, nu, tof, numiter, rtol)

    return coe2rv_hf(k, p, ecc, inc, raan, argp, nu)


@gjit(
    "void(f,f[:],f[:],f,i8,f,f[:],f[:])",
    "(),(n),(n),(),(),()->(n),(n)",
)
def danby_rv_gf(k, r0, v0, tof, numiter, rtol, rr, vv):
    """
    Vectorized danby_rv
    """

    (rr[0], rr[1], rr[2]), (vv[0], vv[1], vv[2]) = danby_rv_hf(
        k,
        array_to_V_hf(r0),
        array_to_V_hf(v0),
        tof,
        numiter,
        rtol,
    )
