from math import log, sqrt

from ..elements import coe2rv_hf, rv2coe_hf, RV2COE_TOL
from ..math.linalg import add_VV_hf, matmul_VV_hf, mul_Vs_hf, norm_V_hf, sign_hf
from ..math.special import stumpff_c2_hf, stumpff_c3_hf
from ..jit import array_to_V_hf, hjit, vjit, gjit


__all__ = [
    "vallado_coe_hf",
    "vallado_coe_vf",
    "vallado_rv_hf",
    "vallado_rv_gf",
    "VALLADO_NUMITER",
]


VALLADO_NUMITER = 350


@hjit("Tuple([f,f,f,f])(f,V,V,f,i8)")
def _vallado_hf(k, r0, v0, tof, numiter):
    r"""Solves Kepler's Equation by applying a Newton-Raphson method.

    If the position of a body along its orbit wants to be computed
    for a specific time, it can be solved by terms of the Kepler's Equation:

    .. math::
        E = M + e\sin{E}

    In this case, the equation is written in terms of the Universal Anomaly:

    .. math::

        \sqrt{\mu}\Delta t = \frac{r_{o}v_{o}}{\sqrt{\mu}}\chi^{2}C(\alpha \chi^{2}) + (1 - \alpha r_{o})\chi^{3}S(\alpha \chi^{2}) + r_{0}\chi

    This equation is solved for the universal anomaly by applying a Newton-Raphson numerical method.
    Once it is solved, the Lagrange coefficients are returned:

    .. math::

        \begin{align}
            f &= 1 \frac{\chi^{2}}{r_{o}}C(\alpha \chi^{2}) \\
            g &= \Delta t - \frac{1}{\sqrt{\mu}}\chi^{3}S(\alpha \chi^{2}) \\
            \dot{f} &= \frac{\sqrt{\mu}}{rr_{o}}(\alpha \chi^{3}S(\alpha \chi^{2}) - \chi) \\
            \dot{g} &= 1 - \frac{\chi^{2}}{r}C(\alpha \chi^{2}) \\
        \end{align}

    Lagrange coefficients can be related then with the position and velocity vectors:

    .. math::
        \begin{align}
            \vec{r} &= f\vec{r_{o}} + g\vec{v_{o}} \\
            \vec{v} &= \dot{f}\vec{r_{o}} + \dot{g}\vec{v_{o}} \\
        \end{align}

    Parameters
    ----------
    k : float
        Standard gravitational parameter.
    r0 : tuple[float,float,float]
        Initial position vector.
    v0 : tuple[float,float,float]
        Initial velocity vector.
    tof : float
        Time of flight.
    numiter : int
        Number of iterations.

    Returns
    -------
    f: float
        First Lagrange coefficient
    g: float
        Second Lagrange coefficient
    fdot: float
        Derivative of the first coefficient
    gdot: float
        Derivative of the second coefficient

    Notes
    -----
    The theoretical procedure is explained in section 3.7 of Curtis in really
    deep detail. For analytical example, check in the same book for example 3.6.

    """
    # Cache some results
    dot_r0v0 = matmul_VV_hf(r0, v0)
    norm_r0 = norm_V_hf(r0)
    sqrt_mu = k**0.5
    alpha = -matmul_VV_hf(v0, v0) / k + 2 / norm_r0

    # First guess
    if alpha > 0:
        # Elliptic orbit
        xi_new = sqrt_mu * tof * alpha
    elif alpha < 0:
        # Hyperbolic orbit
        xi_new = (
            sign_hf(tof)
            * (-1 / alpha) ** 0.5
            * log(
                (-2 * k * alpha * tof)
                / (dot_r0v0 + sign_hf(tof) * sqrt(-k / alpha) * (1 - norm_r0 * alpha))
            )
        )
    else:
        # Parabolic orbit
        # (Conservative initial guess)
        xi_new = sqrt_mu * tof / norm_r0

    # Newton-Raphson iteration on the Kepler equation
    count = 0
    while count < numiter:
        xi = xi_new
        psi = xi * xi * alpha
        c2_psi = stumpff_c2_hf(psi)
        c3_psi = stumpff_c3_hf(psi)
        norm_r = (
            xi * xi * c2_psi
            + dot_r0v0 / sqrt_mu * xi * (1 - psi * c3_psi)
            + norm_r0 * (1 - psi * c2_psi)
        )
        xi_new = (
            xi
            + (
                sqrt_mu * tof
                - xi * xi * xi * c3_psi
                - dot_r0v0 / sqrt_mu * xi * xi * c2_psi
                - norm_r0 * xi * (1 - psi * c3_psi)
            )
            / norm_r
        )
        if abs(xi_new - xi) < 1e-7:
            break
        else:
            count += 1
    else:
        raise RuntimeError("Maximum number of iterations reached")

    # Compute Lagrange coefficients
    f = 1 - xi**2 / norm_r0 * c2_psi
    g = tof - xi**3 / sqrt_mu * c3_psi

    gdot = 1 - xi**2 / norm_r * c2_psi
    fdot = sqrt_mu / (norm_r * norm_r0) * xi * (psi * c3_psi - 1)

    return f, g, fdot, gdot


@hjit("Tuple([V,V])(f,V,V,f,i8)")
def vallado_rv_hf(k, r0, v0, tof, numiter):
    """
    Scalar vallado_rv
    """

    # Compute Lagrange coefficients
    f, g, fdot, gdot = _vallado_hf(k, r0, v0, tof, numiter)

    assert (
        abs(f * gdot - fdot * g - 1) < 1e-5
    ), "Internal error, solution is not consistent"  # Fixed tolerance

    # Return position and velocity vectors
    r = add_VV_hf(mul_Vs_hf(r0, f), mul_Vs_hf(v0, g))
    v = add_VV_hf(mul_Vs_hf(r0, fdot), mul_Vs_hf(v0, gdot))

    return r, v


@gjit("void(f,f[:],f[:],f,i8,f[:],f[:])", "(),(n),(n),(),()->(n),(n)")
def vallado_rv_gf(k, r0, v0, tof, numiter, rr, vv):
    """
    Vectorized vallado_rv
    """

    (rr[0], rr[1], rr[2]), (vv[0], vv[1], vv[2]) = vallado_rv_hf(
        k, array_to_V_hf(r0), array_to_V_hf(v0), tof, numiter
    )


@hjit("f(f,f,f,f,f,f,f,f,i8)")
def vallado_coe_hf(k, p, ecc, inc, raan, argp, nu, tof, numiter):
    """
    Scalar vallado_coe
    """

    r0, v0 = coe2rv_hf(k, p, ecc, inc, raan, argp, nu)
    rr, vv = vallado_rv_hf(k, r0, v0, tof, numiter)
    _, _, _, _, _, nu_ = rv2coe_hf(k, rr, vv, RV2COE_TOL)

    return nu_


@vjit("f(f,f,f,f,f,f,f,f,i8)")
def vallado_coe_vf(k, p, ecc, inc, raan, argp, nu, tof, numiter):
    """
    Vectorized vallado_coe
    """

    return vallado_coe_hf(k, p, ecc, inc, raan, argp, nu, tof, numiter)
