from math import cos, sin, sqrt

from ..angles import E_to_M_hf, E_to_nu_hf, nu_to_E_hf
from ..elements import coe2rv_hf, rv2coe_hf, RV2COE_TOL
from ..jit import array_to_V_hf, hjit, vjit, gjit


__all__ = [
    "gooding_coe_hf",
    "gooding_coe_vf",
    "gooding_rv_hf",
    "gooding_rv_gf",
    "GOODING_NUMITER",
    "GOODING_RTOL",
]


GOODING_NUMITER = 150
GOODING_RTOL = 1e-8


@hjit("f(f,f,f,f,f,f,f,f,i8,f)")
def gooding_coe_hf(k, p, ecc, inc, raan, argp, nu, tof, numiter, rtol):
    """
    Scalar gooding_coe
    """
    # TODO: parabolic and hyperbolic not implemented cases
    if ecc >= 1.0:
        raise NotImplementedError(
            "Parabolic/Hyperbolic cases still not implemented in gooding."
        )

    M0 = E_to_M_hf(nu_to_E_hf(nu, ecc), ecc)
    semi_axis_a = p / (1 - ecc**2)
    n = sqrt(k / abs(semi_axis_a) ** 3)
    M = M0 + n * tof

    # Start the computation
    n = 0
    c = ecc * cos(M)
    s = ecc * sin(M)
    psi = s / sqrt(1 - 2 * c + ecc**2)
    f = 1.0
    while f**2 >= rtol and n <= numiter:
        xi = cos(psi)
        eta = sin(psi)
        fd = (1 - c * xi) + s * eta
        fdd = c * eta + s * xi
        f = psi - fdd
        psi = psi - f * fd / (fd**2 - 0.5 * f * fdd)
        n += 1

    E = M + psi
    return E_to_nu_hf(E, ecc)


@vjit("f(f,f,f,f,f,f,f,f,i8,f)")
def gooding_coe_vf(k, p, ecc, inc, raan, argp, nu, tof, numiter, rtol):
    """
    Vectorized gooding_coe
    """

    return gooding_coe_hf(k, p, ecc, inc, raan, argp, nu, tof, numiter, rtol)


@hjit("Tuple([V,V])(f,V,V,f,i8,f)")
def gooding_rv_hf(k, r0, v0, tof, numiter, rtol):
    """Solves the Elliptic Kepler Equation with a cubic convergence and
    accuracy better than 10e-12 rad is normally achieved. It is not valid for
    eccentricities equal or higher than 1.0.

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
        Number of iterations, defaults to 150.
    rtol : float, optional
        Relative error for accuracy of the method, defaults to 1e-8.

    Returns
    -------
    rr : tuple[float,float,float]
        Final position vector.
    vv : tuple[float,float,float]
        Final velocity vector.
    Note
    ----
    Original paper for the algorithm: https://doi.org/10.1007/BF01238923
    """
    # Solve first for eccentricity and mean anomaly
    p, ecc, inc, raan, argp, nu = rv2coe_hf(k, r0, v0, RV2COE_TOL)
    nu = gooding_coe_hf(k, p, ecc, inc, raan, argp, nu, tof, numiter, rtol)

    return coe2rv_hf(k, p, ecc, inc, raan, argp, nu)


@gjit("void(f,f[:],f[:],f,i8,f,f[:],f[:])", "(),(n),(n),(),(),()->(),()")
def gooding_rv_gf(k, r0, v0, tof, numiter, rtol, rr, vv):
    """
    Vectorized gooding_rv
    """

    (rr[0], rr[1], rr[2]), (vv[0], vv[1], vv[2]) = gooding_rv_hf(
        k, array_to_V_hf(r0), array_to_V_hf(v0), tof, numiter, rtol
    )
