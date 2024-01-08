from math import floor, pi, sin, sqrt

from ..angles import E_to_M_hf, E_to_nu_hf, nu_to_E_hf
from ..elements import coe2rv_hf, rv2coe_hf, RV2COE_TOL
from ..jit import array_to_V_hf, hjit, vjit, gjit


__all__ = [
    "recseries_coe_hf",
    "recseries_coe_vf",
    "recseries_rv_hf",
    "recseries_rv_gf",
    "RECSERIES_METHOD_RTOL",
    "RECSERIES_METHOD_ORDER",
    "RECSERIES_ORDER",
    "RECSERIES_NUMITER",
    "RECSERIES_RTOL",
]


RECSERIES_METHOD_RTOL = 0
RECSERIES_METHOD_ORDER = 1
RECSERIES_ORDER = 8
RECSERIES_NUMITER = 100
RECSERIES_RTOL = 1e-8


@hjit("f(f,f,f,f,f,f,f,f,i8,i8,i8,f)")
def recseries_coe_hf(
    k,
    p,
    ecc,
    inc,
    raan,
    argp,
    nu,
    tof,
    method,
    order,
    numiter,
    rtol,
):
    """
    Scalar recseries_coe
    """

    # semi-major axis
    semi_axis_a = p / (1 - ecc**2)
    # mean angular motion
    n = sqrt(k / abs(semi_axis_a) ** 3)

    if ecc == 0:
        # Solving for circular orbit

        # compute initial mean anoamly
        M0 = nu  # For circular orbit (M = E = nu)
        # final mean anaomaly
        M = M0 + n * tof
        # snapping anomaly to [0,pi] range
        nu = M - 2 * pi * floor(M / 2 / pi)

        return nu

    elif ecc < 1.0:
        # Solving for elliptical orbit

        # compute initial mean anoamly
        M0 = E_to_M_hf(nu_to_E_hf(nu, ecc), ecc)
        # final mean anaomaly
        M = M0 + n * tof
        # snapping anomaly to [0,pi] range
        M = M - 2 * pi * floor(M / 2 / pi)

        # set recursion iteration
        if method == RECSERIES_METHOD_RTOL:
            Niter = numiter
        elif method == RECSERIES_METHOD_ORDER:
            Niter = order
        else:
            raise ValueError("Unknown recursion termination method ('rtol','order').")

        # compute eccentric anomaly through recursive series
        E = M + ecc  # Using initial guess from vallado to improve convergence
        for i in range(0, Niter):
            En = M + ecc * sin(E)
            # check for break condition
            if method == "rtol" and (abs(En - E) / abs(E)) < rtol:
                break
            E = En

        return E_to_nu_hf(E, ecc)

    else:
        # Parabolic/Hyperbolic orbits are not supported
        raise ValueError("Parabolic/Hyperbolic orbits not supported.")

    return nu


@vjit("f(f,f,f,f,f,f,f,f,i8,i8,i8,f)")
def recseries_coe_vf(
    k,
    p,
    ecc,
    inc,
    raan,
    argp,
    nu,
    tof,
    method,
    order,
    numiter,
    rtol,
):
    """
    Vectorized recseries_coe
    """

    return recseries_coe_hf(
        k,
        p,
        ecc,
        inc,
        raan,
        argp,
        nu,
        tof,
        method,
        order,
        numiter,
        rtol,
    )


@hjit("Tuple([V,V])(f,V,V,f,i8,i8,i8,f)")
def recseries_rv_hf(k, r0, v0, tof, method, order, numiter, rtol):
    """Kepler solver for elliptical orbits with recursive series approximation
    method. The order of the series is a user defined parameter.

    Parameters
    ----------
    k : float
        Standard gravitational parameter of the attractor.
    r0 : numpy.ndarray
        Position vector.
    v0 : numpy.ndarray
        Velocity vector.
    tof : float
        Time of flight.
    method : str
        Type of termination method ('rtol','order')
    order : int, optional
        Order of recursion, defaults to 8.
    numiter : int, optional
        Number of iterations, defaults to 100.
    rtol : float, optional
        Relative error for accuracy of the method, defaults to 1e-8.

    Returns
    -------
    rr : numpy.ndarray
        Final position vector.
    vv : numpy.ndarray
        Final velocity vector.

    Notes
    -----
    This algorithm uses series discussed in the paper *Recursive solution to
    Kepler’s problem for elliptical orbits - application in robust
    Newton-Raphson and co-planar closest approach estimation*
    with DOI: http://dx.doi.org/10.13140/RG.2.2.18578.58563/1
    """
    # Solve first for eccentricity and mean anomaly
    p, ecc, inc, raan, argp, nu = rv2coe_hf(k, r0, v0, RV2COE_TOL)
    nu = recseries_coe_hf(
        k, p, ecc, inc, raan, argp, nu, tof, method, order, numiter, rtol
    )

    return coe2rv_hf(k, p, ecc, inc, raan, argp, nu)


@gjit("void(f,f[:],f[:],f,i8,i8,i8,f,f[:],f[:])", "(),(n),(n),(),(),(),(),()->(n),(n)")
def recseries_rv_gf(k, r0, v0, tof, method, order, numiter, rtol, rr, vv):
    """
    Vectorized recseries_rv
    """

    (rr[0], rr[1], rr[2]), (vv[0], vv[1], vv[2]) = recseries_rv_hf(
        k, array_to_V_hf(r0), array_to_V_hf(v0), tof, method, order, numiter, rtol
    )
