from math import exp, pow as pow_

from .events import line_of_sight_hf
from .jit import hjit
from .math.linalg import norm_V_hf, mul_Vs_hf, mul_VV_hf, sub_VV_hf


__all__ = [
    "J2_perturbation_hf",
    "J3_perturbation_hf",
    "atmospheric_drag_exponential_hf",
    "atmospheric_drag_hf",
    "third_body_hf",
    "radiation_pressure_hf",
]


@hjit("V(f,V,V,f,f,f)")
def J2_perturbation_hf(t0, rr, vv, k, J2, R):
    r"""Calculates J2_perturbation acceleration (km/s2).

    .. math::

        \vec{p} = \frac{3}{2}\frac{J_{2}\mu R^{2}}{r^{4}}\left [\frac{x}{r}\left ( 5\frac{z^{2}}{r^{2}}-1 \right )\vec{i} + \frac{y}{r}\left ( 5\frac{z^{2}}{r^{2}}-1 \right )\vec{j} + \frac{z}{r}\left ( 5\frac{z^{2}}{r^{2}}-3 \right )\vec{k}\right]

    .. versionadded:: 0.9.0

    Parameters
    ----------
    t0 : float
        Current time (s)
    rr : tuple[float,float,float]
        Vector [x, y, z] (km)
    vv : tuple[float,float,float]
        Vector [vx, vy, vz] (km/s)
    k : float
        Standard Gravitational parameter. (km^3/s^2)
    J2 : float
        Oblateness factor
    R : float
        Attractor radius

    Notes
    -----
    The J2 accounts for the oblateness of the attractor. The formula is given in
    Howard Curtis, (12.30)

    """
    r = norm_V_hf(rr)

    factor = 1.5 * k * J2 * R * R / pow_(r, 5)
    a_base = 5.0 * rr[2] * rr[2] / (r * r)
    a = a_base - 1, a_base - 1, a_base - 3
    return mul_Vs_hf(mul_VV_hf(a, rr), factor)


@hjit("V(f,V,V,f,f,f)")
def J3_perturbation_hf(t0, rr, vv, k, J3, R):
    r"""Calculates J3_perturbation acceleration (km/s2).

    Parameters
    ----------
    t0 : float
        Current time (s)
    rr : tuple[float,float,float]
        Vector [x, y, z] (km)
    vv : tuple[float,float,float]
        Vector [vx, vy, vz] (km/s)
    k : float
        Standard Gravitational parameter. (km^3/s^2)
    J3 : float
        Oblateness factor
    R : float
        Attractor radius

    Notes
    -----
    The J3 accounts for the oblateness of the attractor. The formula is given in
    Howard Curtis, problem 12.8
    This perturbation has not been fully validated, see https://github.com/poliastro/poliastro/pull/398

    """
    r = norm_V_hf(rr)

    factor = (1.0 / 2.0) * k * J3 * (R**3) / (r**5)
    cos_phi = rr[2] / r

    a_x = 5.0 * rr[0] / r * (7.0 * cos_phi**3 - 3.0 * cos_phi)
    a_y = 5.0 * rr[1] / r * (7.0 * cos_phi**3 - 3.0 * cos_phi)
    a_z = 3.0 * (35.0 / 3.0 * cos_phi**4 - 10.0 * cos_phi**2 + 1)
    return a_x * factor, a_y * factor, a_z * factor


@hjit("V(f,V,V,f,f,f,f,f,f)")
def atmospheric_drag_exponential_hf(t0, rr, vv, k, R, C_D, A_over_m, H0, rho0):
    r"""Calculates atmospheric drag acceleration (km/s2).

    .. math::

        \vec{p} = -\frac{1}{2}\rho v_{rel}\left ( \frac{C_{d}A}{m} \right )\vec{v_{rel}}

    .. versionadded:: 0.9.0

    Parameters
    ----------
    t0 : float
        Current time (s)
    rr : tuple[float,float,float]
        Vector [x, y, z] (km)
    vv : tuple[float,float,float]
        Vector [vx, vy, vz] (km/s)
    k : float
        Standard Gravitational parameter (km^3/s^2).
    R : float
        Radius of the attractor (km)
    C_D : float
        Dimensionless drag coefficient ()
    A_over_m : float
        Frontal area/mass of the spacecraft (km^2/kg)
    H0 : float
        Atmospheric scale height, (km)
    rho0 : float
        Exponent density pre-factor, (kg / km^3)

    Notes
    -----
    This function provides the acceleration due to atmospheric drag
    using an overly-simplistic exponential atmosphere model. We follow
    Howard Curtis, section 12.4
    the atmospheric density model is rho(H) = rho0 x exp(-H / H0)

    """
    H = norm_V_hf(rr)

    v = norm_V_hf(vv)
    B = C_D * A_over_m
    rho = rho0 * exp(-(H - R) / H0)

    return mul_Vs_hf(vv, -(1.0 / 2.0) * rho * B * v)


@hjit("V(f,V,V,f,f,f,f)")
def atmospheric_drag_hf(t0, rr, vv, k, C_D, A_over_m, rho):
    r"""Calculates atmospheric drag acceleration (km/s2).

    .. math::

        \vec{p} = -\frac{1}{2}\rho v_{rel}\left ( \frac{C_{d}A}{m} \right )\vec{v_{rel}}

    .. versionadded:: 1.14

    Parameters
    ----------
    t0 : float
        Current time (s).
    rr : tuple[float,float,float]
        Vector [x, y, z] (km)
    vv : tuple[float,float,float]
        Vector [vx, vy, vz] (km/s)
    k : float
        Standard Gravitational parameter (km^3/s^2)
    C_D : float
        Dimensionless drag coefficient ()
    A_over_m : float
        Frontal area/mass of the spacecraft (km^2/kg)
    rho : float
        Air density at corresponding state (kg/km^3)

    Notes
    -----
    This function provides the acceleration due to atmospheric drag, as
    computed by a model from hapsira.earth.atmosphere

    """
    v = norm_V_hf(vv)
    B = C_D * A_over_m

    return mul_Vs_hf(vv, -(1.0 / 2.0) * rho * B * v)


@hjit("V(f,V,V,f,f,F(V(f)))")
def third_body_hf(t0, rr, vv, k, k_third, perturbation_body):
    r"""Calculate third body acceleration (km/s2).

    .. math::

        \vec{p} = \mu_{m}\left ( \frac{\vec{r_{m/s}}}{r_{m/s}^3} - \frac{\vec{r_{m}}}{r_{m}^3} \right )

    Parameters
    ----------
    t0 : float
        Current time (s).
    rr : tuple[float,float,float]
        Vector [x, y, z] (km)
    vv : tuple[float,float,float]
        Vector [vx, vy, vz] (km/s)
    k : float
        Standard Gravitational parameter of the attractor (km^3/s^2).
    k_third : float
        Standard Gravitational parameter of the third body (km^3/s^2).
    perturbation_body : callable
        A callable object returning the position of the body that causes the perturbation
        in the attractor frame.

    Notes
    -----
    This formula is taken from Howard Curtis, section 12.10. As an example, a third body could be
    the gravity from the Moon acting on a small satellite.

    """
    body_r = perturbation_body(t0)
    delta_r = sub_VV_hf(body_r, rr)
    return sub_VV_hf(
        mul_Vs_hf(delta_r, k_third / norm_V_hf(delta_r) ** 3),
        mul_Vs_hf(body_r, k_third / norm_V_hf(body_r) ** 3),
    )


@hjit("V(f,V,V,f,f,f,f,f,F(V(f)))")
def radiation_pressure_hf(t0, rr, vv, k, R, C_R, A_over_m, Wdivc_s, star):
    r"""Calculates radiation pressure acceleration (km/s2).

    .. math::

        \vec{p} = -\nu \frac{S}{c} \left ( \frac{C_{r}A}{m} \right )\frac{\vec{r}}{r}

    Parameters
    ----------
    t0 : float
        Current time (s).
    rr : tuple[float,float,float]
        Vector [x, y, z] (km)
    vv : tuple[float,float,float]
        Vector [vx, vy, vz] (km/s)
    k : float
        Standard Gravitational parameter (km^3/s^2).
    R : float
        Radius of the attractor.
    C_R : float
        Dimensionless radiation pressure coefficient, 1 < C_R < 2 ().
    A_over_m : float
        Effective spacecraft area/mass of the spacecraft (km^2/kg).
    Wdivc_s : float
        Total star emitted power divided by the speed of light (kg km/s^2).
    star : callable
        A callable object returning the position of radiating star
        in the attractor frame.

    Notes
    -----
    This function provides the acceleration due to star light pressure. We follow
    Howard Curtis, section 12.9

    """
    r_star = star(t0)
    P_s = Wdivc_s / (norm_V_hf(r_star) ** 2)

    if line_of_sight_hf(rr, r_star, R) > 0:
        nu = 1.0
    else:
        nu = 0.0
    return mul_Vs_hf(r_star, -nu * P_s * (C_R * A_over_m) / norm_V_hf(r_star))
