from math import atan2, cos, pi, sin, tan

from ..jit import hjit, gjit
from ..elements import circular_velocity_hf
from ..math.linalg import (
    add_VV_hf,
    cross_VV_hf,
    div_Vs_hf,
    mul_Vs_hf,
    norm_V_hf,
    sign_hf,
)


__all__ = [
    "change_a_inc_hb",
]


@hjit("f(f,f,f,f)")
def _beta_0_hf(V_0, V_f, inc_0, inc_f):
    """Compute initial yaw angle (β) as a function of the problem parameters."""
    delta_i_f = abs(inc_f - inc_0)
    return atan2(
        sin(pi / 2 * delta_i_f),
        V_0 / V_f - cos(pi / 2 * delta_i_f),
    )


@hjit("Tuple([f,f,f])(f,f,f,f,f)")
def _compute_parameters_hf(k, a_0, a_f, inc_0, inc_f):
    """Compute parameters of the model."""
    V_0 = circular_velocity_hf(k, a_0)
    V_f = circular_velocity_hf(k, a_f)
    beta_0_ = _beta_0_hf(V_0, V_f, inc_0, inc_f)

    return V_0, V_f, beta_0_


@gjit("void(f,f,f,f,f,f[:],f[:],f[:])", "(),(),(),(),()->(),(),()")
def _compute_parameters_gf(k, a_0, a_f, inc_0, inc_f, V_0, V_f, beta_0_):
    """
    Vectorized compute_parameters
    """

    V_0[0], V_f[0], beta_0_[0] = _compute_parameters_hf(k, a_0, a_f, inc_0, inc_f)


@hjit("f(f,f,f,f,f)")
def _delta_V_hf(V_0, V_f, beta_0, inc_0, inc_f):
    """Compute required increment of velocity."""
    delta_i_f = abs(inc_f - inc_0)
    if delta_i_f == 0:
        return abs(V_f - V_0)
    return V_0 * cos(beta_0) - V_0 * sin(beta_0) / tan(pi / 2 * delta_i_f + beta_0)


@hjit("Tuple([f,f])(f,f,f,f,f,f)")
def _extra_quantities_hf(k, a_0, a_f, inc_0, inc_f, f):
    """Extra quantities given by the Edelbaum (a, i) model."""
    V_0, V_f, beta_0_ = _compute_parameters_hf(k, a_0, a_f, inc_0, inc_f)
    delta_V = _delta_V_hf(V_0, V_f, beta_0_, inc_0, inc_f)
    t_f_ = delta_V / f

    return delta_V, t_f_


@gjit("void(f,f,f,f,f,f,f[:],f[:])", "(),(),(),(),(),()->(),()")
def _extra_quantities_gf(k, a_0, a_f, inc_0, inc_f, f, delta_V, t_f_):
    """
    Vectorized extra_quantities
    """

    delta_V[0], t_f_[0] = _extra_quantities_hf(k, a_0, a_f, inc_0, inc_f, f)


@hjit("f(f,f,f,f)")
def _beta_hf(t, V_0, f, beta_0):
    """Compute yaw angle (β) as a function of time and the problem parameters."""
    return atan2(V_0 * sin(beta_0), V_0 * cos(beta_0) - f * t)


def change_a_inc_hb(k, a_0, a_f, inc_0, inc_f, f):
    """Change semimajor axis and inclination.
       Guidance law from the Edelbaum/Kéchichian theory, optimal transfer between circular inclined orbits
       (a_0, i_0) --> (a_f, i_f), ecc = 0.

    Parameters
    ----------
    k : float
        Gravitational parameter.
    a_0 : float
        Initial semimajor axis (km).
    a_f : float
        Final semimajor axis (km).
    inc_0 : float
        Initial inclination (rad).
    inc_f : float
        Final inclination (rad).
    f : float
        Magnitude of constant acceleration (km / s**2).

    Returns
    -------
    a_d : function
    delta_V : float
    t_f : float

    Notes
    -----
    Edelbaum theory, reformulated by Kéchichian.

    References
    ----------
    * Edelbaum, T. N. "Propulsion Requirements delta_V for Controllable
      Satellites", 1961.
    * Kéchichian, J. A. "Reformulation of Edelbaum's Low-Thrust
      Transfer Problem Using Optimal Control Theory", 1997.
    """
    V_0, _, beta_0_ = _compute_parameters_gf(  # pylint: disable=E1120,E0633
        k, a_0, a_f, inc_0, inc_f
    )

    @hjit("V(f,V,V,f)", cache=False)
    def a_d_hf(t0, rr, vv, k):
        # Change sign of beta with the out-of-plane velocity
        beta_ = _beta_hf(t0, V_0, f, beta_0_) * sign_hf(rr[0] * (inc_f - inc_0))

        t_ = div_Vs_hf(vv, norm_V_hf(vv))
        crv = cross_VV_hf(rr, vv)
        w_ = div_Vs_hf(crv, norm_V_hf(crv))
        accel_v = mul_Vs_hf(
            add_VV_hf(mul_Vs_hf(t_, cos(beta_)), mul_Vs_hf(w_, sin(beta_))), f
        )
        return accel_v

    delta_V, t_f = _extra_quantities_gf(  # pylint: disable=E1120,E0633
        k, a_0, a_f, inc_0, inc_f, f
    )
    return a_d_hf, delta_V, t_f
