from math import cos, pi, sin, sqrt

from ..elements import circular_velocity_hf, rv2coe_hf, RV2COE_TOL
from ..jit import hjit, gjit
from ..math.linalg import add_VV_hf, cross_VV_hf, div_Vs_hf, mul_Vs_hf, norm_hf, sign_hf


__all__ = [
    "change_argp_hb",
]


@hjit("f(f,f,f,f,f,f)")
def _delta_V_hf(V, ecc, argp_0, argp_f, f, A):
    """Compute required increment of velocity."""
    delta_argp = argp_f - argp_0
    return delta_argp / (
        3 * sign_hf(delta_argp) / 2 * sqrt(1 - ecc**2) / ecc / V + A / f
    )


@hjit("Tuple([f,f])(f,f,f,f,f,f,f)")
def _extra_quantities_hf(k, a, ecc, argp_0, argp_f, f, A):
    """Extra quantities given by the model."""
    V = circular_velocity_hf(k, a)
    delta_V_ = _delta_V_hf(V, ecc, argp_0, argp_f, f, A)
    t_f_ = delta_V_ / f

    return delta_V_, t_f_


@gjit("void(f,f,f,f,f,f,f,f[:],f[:])", "(),(),(),(),(),(),()->(),()")
def _extra_quantities_gf(k, a, ecc, argp_0, argp_f, f, A, delta_V_, t_f_):
    """
    Vectorized extra_quantities
    """

    delta_V_[0], t_f_[0] = _extra_quantities_hf(k, a, ecc, argp_0, argp_f, f, A)


def change_argp_hb(k, a, ecc, argp_0, argp_f, f):
    """Guidance law from the model.
    Thrust is aligned with an inertially fixed direction perpendicular to the
    semimajor axis of the orbit.

    Parameters
    ----------
    k : float
        Gravitational parameter (km**3 / s**2)
    a : float
        Semi-major axis (km)
    ecc : float
        Eccentricity
    argp_0 : float
        Initial argument of periapsis (rad)
    argp_f : float
        Final argument of periapsis (rad)
    f : float
        Magnitude of constant acceleration (km / s**2)

    Returns
    -------
    a_d : function
    delta_V : float
    t_f : float
    """

    @hjit("V(f,V,V,f)")
    def a_d_hf(t0, rr, vv, k):
        nu = rv2coe_hf(k, rr, vv, RV2COE_TOL)[-1]

        alpha_ = nu - pi / 2

        r_ = div_Vs_hf(rr, norm_hf(rr))
        crv = cross_VV_hf(rr, vv)
        w_ = div_Vs_hf(crv, norm_hf(crv))
        s_ = cross_VV_hf(w_, r_)
        accel_v = mul_Vs_hf(
            add_VV_hf(mul_Vs_hf(s_, cos(alpha_)), mul_Vs_hf(r_, sin(alpha_))), f
        )
        return accel_v

    delta_V, t_f = _extra_quantities_gf(  # pylint: disable=E1120,E0633
        k, a, ecc, argp_0, argp_f, f, 0.0
    )

    return a_d_hf, delta_V, t_f
