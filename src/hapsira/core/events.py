from math import acos, asin, cos, sin, sqrt

from .elements import coe_rotation_matrix_hf, rv2coe_hf, RV2COE_TOL
from .jit import array_to_V_hf, hjit, gjit
from .math.linalg import div_Vs_hf, matmul_MV_hf, matmul_VV_hf, norm_V_hf, sub_VV_hf
from .util import planetocentric_to_AltAz_hf


__all__ = [
    "ECLIPSE_UMBRA",
    "eclipse_function_hf",
    "eclipse_function_gf",
    "line_of_sight_hf",
    "line_of_sight_gf",
    "elevation_function_hf",
    "elevation_function_gf",
]


ECLIPSE_UMBRA = True


@hjit("f(f,V,V,V,f,f,b1)")
def eclipse_function_hf(k, rr, vv, r_sec, R_sec, R_primary, umbra):
    """Calculates a continuous shadow function.

    Parameters
    ----------
    k : float
        Standard gravitational parameter (km^3 / s^2).
    rr : tuple[float,float,float]
        Satellite position vector with respect to the primary body.
    vv : tuple[float,float,float]
        Satellite velocity vector with respect to the primary body.
    r_sec : tuple[float,float,float]
        Position vector of the secondary body with respect to the primary body.
    R_sec : float
        Equatorial radius of the secondary body.
    R_primary : float
        Equatorial radius of the primary body.
    umbra : bool
        Whether to calculate the shadow function for umbra or penumbra, defaults to True
        i.e. calculates for umbra.

    Notes
    -----
    The shadow function is taken from Escobal, P. (1985). Methods of orbit determination.
    The current implementation assumes circular bodies and doesn't account for flattening.

    """

    # Plus or minus condition
    pm = 1 if umbra else -1
    p, ecc, inc, raan, argp, nu = rv2coe_hf(k, rr, vv, RV2COE_TOL)

    PQW = coe_rotation_matrix_hf(inc, raan, argp)
    P_ = PQW[0][0], PQW[1][0], PQW[2][0]
    Q_ = PQW[0][1], PQW[1][1], PQW[2][1]

    r_sec_norm = norm_V_hf(r_sec)
    beta = matmul_VV_hf(P_, r_sec) / r_sec_norm
    zeta = matmul_VV_hf(Q_, r_sec) / r_sec_norm

    sin_delta_shadow = sin((R_sec - pm * R_primary) / r_sec_norm)

    cos_psi = beta * cos(nu) + zeta * sin(nu)
    shadow_function = (
        ((R_primary**2) * (1 + ecc * cos(nu)) ** 2)
        + (p**2) * (cos_psi**2)
        - p**2
        + pm * (2 * p * R_primary * cos_psi) * (1 + ecc * cos(nu)) * sin_delta_shadow
    )

    return shadow_function


@gjit(
    "void(f,f[:],f[:],f[:],f,f,b1,f[:])",
    "(),(n),(n),(n),(),(),()->()",
)
def eclipse_function_gf(k, rr, vv, r_sec, R_sec, R_primary, umbra, eclipse):
    """
    Vectorized eclipse_function
    """

    eclipse[0] = eclipse_function_hf(
        k,
        array_to_V_hf(rr),
        array_to_V_hf(vv),
        array_to_V_hf(r_sec),
        R_sec,
        R_primary,
        umbra,
    )


@hjit("f(V,V,f)")
def line_of_sight_hf(r1, r2, R):
    """Calculates the line of sight condition between two position vectors, r1 and r2.

    Parameters
    ----------
    r1 : tuple[float,float,float]
        The position vector of the first object with respect to a central attractor.
    r2 : tuple[float,float,float]
        The position vector of the second object with respect to a central attractor.
    R : float
        The radius of the central attractor.

    Returns
    -------
    delta_theta: float
        Greater than or equal to zero, if there exists a LOS between two objects
        located by r1 and r2, else negative.

    """
    r1_norm = norm_V_hf(r1)
    r2_norm = norm_V_hf(r2)

    theta = acos(matmul_VV_hf(r1, r2) / r1_norm / r2_norm)
    theta_1 = acos(R / r1_norm)
    theta_2 = acos(R / r2_norm)

    return (theta_1 + theta_2) - theta


@gjit("void(f[:],f[:],f,f[:])", "(n),(n),()->()")
def line_of_sight_gf(r1, r2, R, delta_theta):
    """
    Vectorized line_of_sight
    """

    delta_theta[0] = line_of_sight_hf(array_to_V_hf(r1), array_to_V_hf(r2), R)


@hjit("f(V,f,f,f,f,f)")
def elevation_function_hf(rr, phi, theta, R, R_p, H):
    """Calculates the elevation angle of an object in orbit with respect to
    a location on attractor.

    Parameters
    ----------
    rr: tuple[float,float,float]
        Satellite position vector with respect to the central attractor.
    phi: float
        Geodetic Latitude of the station.
    theta: float
        Local sidereal time at a particular instant.
    R: float
        Equatorial radius of the central attractor.
    R_p: float
        Polar radius of the central attractor.
    H: float
        Elevation, above the ellipsoidal surface.
    """

    cos_phi = cos(phi)
    sin_phi = sin(phi)

    ecc = sqrt(1 - (R_p / R) ** 2)
    denom = sqrt(1 - ecc * ecc * sin_phi * sin_phi)
    g1 = H + (R / denom)
    g2 = H + (1 - ecc * ecc) * R / denom

    # Coordinates of location on attractor.
    coords = (
        g1 * cos_phi * cos(theta),
        g1 * cos_phi * sin(theta),
        g2 * sin_phi,
    )

    # Position of satellite with respect to a point on attractor.
    rho = sub_VV_hf(rr, coords)

    rot_matrix = planetocentric_to_AltAz_hf(theta, phi)

    new_rho = matmul_MV_hf(rot_matrix, rho)
    new_rho = div_Vs_hf(new_rho, norm_V_hf(new_rho))
    el = asin(new_rho[-1])

    return el


@gjit("void(f[:],f,f,f,f,f,f[:])", "(n),(),(),(),(),()->()")
def elevation_function_gf(rr, phi, theta, R, R_p, H, el):
    """
    Vectorized elevation_function
    """

    el[0] = elevation_function_hf(array_to_V_hf(rr), phi, theta, R, R_p, H)
