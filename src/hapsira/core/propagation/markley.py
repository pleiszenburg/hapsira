from numba import njit as jit
import numpy as np

from hapsira.core.angles import (
    E_to_M_hf,
    E_to_nu_hf,
    kepler_equation_hf,
    kepler_equation_prime_hf,
    nu_to_E_hf,
)
from hapsira.core.elements import coe2rv_hf, rv2coe_hf, RV2COE_TOL
from ..jit import array_to_V_hf


@jit
def markley_coe(k, p, ecc, inc, raan, argp, nu, tof):
    M0 = E_to_M_hf(nu_to_E_hf(nu, ecc), ecc)
    a = p / (1 - ecc**2)
    n = np.sqrt(k / a**3)
    M = M0 + n * tof

    # Range between -pi and pi
    M = (M + np.pi) % (2 * np.pi) - np.pi

    # Equation (20)
    alpha = (3 * np.pi**2 + 1.6 * (np.pi - np.abs(M)) / (1 + ecc)) / (np.pi**2 - 6)

    # Equation (5)
    d = 3 * (1 - ecc) + alpha * ecc

    # Equation (9)
    q = 2 * alpha * d * (1 - ecc) - M**2

    # Equation (10)
    r = 3 * alpha * d * (d - 1 + ecc) * M + M**3

    # Equation (14)
    w = (np.abs(r) + np.sqrt(q**3 + r**2)) ** (2 / 3)

    # Equation (15)
    E = (2 * r * w / (w**2 + w * q + q**2) + M) / d

    # Equation (26)
    f0 = kepler_equation_hf(E, M, ecc)
    f1 = kepler_equation_prime_hf(E, M, ecc)
    f2 = ecc * np.sin(E)
    f3 = ecc * np.cos(E)
    f4 = -f2

    # Equation (22)
    delta3 = -f0 / (f1 - 0.5 * f0 * f2 / f1)
    delta4 = -f0 / (f1 + 0.5 * delta3 * f2 + 1 / 6 * delta3**2 * f3)
    delta5 = -f0 / (
        f1 + 0.5 * delta4 * f2 + 1 / 6 * delta4**2 * f3 + 1 / 24 * delta4**3 * f4
    )

    E += delta5
    nu = E_to_nu_hf(E, ecc)

    return nu


@jit
def markley(k, r0, v0, tof):
    """Solves the kepler problem by a non-iterative method. Relative error is
    around 1e-18, only limited by machine double-precision errors.

    Parameters
    ----------
    k : float
        Standar Gravitational parameter.
    r0 : numpy.ndarray
        Initial position vector wrt attractor center.
    v0 : numpy.ndarray
        Initial velocity vector.
    tof : float
        Time of flight.

    Returns
    -------
    rr: numpy.ndarray
        Final position vector.
    vv: numpy.ndarray
        Final velocity vector.

    Notes
    -----
    The following algorithm was taken from http://dx.doi.org/10.1007/BF00691917.

    """
    # Solve first for eccentricity and mean anomaly
    p, ecc, inc, raan, argp, nu = rv2coe_hf(
        k, array_to_V_hf(r0), array_to_V_hf(v0), RV2COE_TOL
    )
    nu = markley_coe(k, p, ecc, inc, raan, argp, nu, tof)

    return np.array(coe2rv_hf(k, p, ecc, inc, raan, argp, nu))
