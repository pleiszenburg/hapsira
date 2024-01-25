"""Simultaneous eccentricity and inclination changes.

References
----------
* Pollard, J. E. "Simplified Analysis of Low-Thrust Orbital Maneuvers", 2000.
rv2coe
"""

from math import asin, atan, cos, pi, log, sin

from numpy import array

from ..elements import (
    circular_velocity_hf,
    eccentricity_vector_hf,
    rv2coe_hf,
    RV2COE_TOL,
)
from ..jit import array_to_V_hf, hjit, vjit, gjit
from ..math.linalg import (
    add_VV_hf,
    cross_VV_hf,
    div_Vs_hf,
    mul_Vs_hf,
    norm_V_hf,
    sign_hf,
)


__all__ = [
    "beta_hf",
    "beta_vf",
    "change_ecc_inc_hb",
]


@hjit("f(f,f,f,f,f)")
def beta_hf(ecc_0, ecc_f, inc_0, inc_f, argp):
    """
    Scalar beta
    """
    # Note: "The argument of perigee will vary during the orbit transfer
    # due to the natural drift and because e may approach zero.
    # However, [the equation] still gives a good estimate of the desired
    # thrust angle."
    return atan(
        abs(
            3
            * pi
            * (inc_f - inc_0)
            / (
                4
                * cos(argp)
                * (
                    ecc_0
                    - ecc_f
                    + log((1 + ecc_f) * (-1 + ecc_0) / ((1 + ecc_0) * (-1 + ecc_f)))
                )
            )
        )
    )


@vjit("f(f,f,f,f,f)")
def beta_vf(ecc_0, ecc_f, inc_0, inc_f, argp):
    """
    Vectorized beta
    """

    return beta_hf(ecc_0, ecc_f, inc_0, inc_f, argp)


@hjit("f(f,f,f,f)")
def _delta_V_hf(V_0, ecc_0, ecc_f, beta_):
    """
    Compute required increment of velocity.
    """

    return 2 * V_0 * abs(asin(ecc_0) - asin(ecc_f)) / (3 * cos(beta_))


@hjit("f(f,f)")
def _delta_t_hf(delta_v, f):
    """
    Compute required increment of velocity.
    """
    return delta_v / f


@hjit("Tuple([V,f,f,f])(f,f,f,f,f,f,f,V,V,f)")
def _prepare_hf(k, a, ecc_0, ecc_f, inc_0, inc_f, argp, r, v, f):
    """
    Vectorized prepare
    """

    # We fix the inertial direction at the beginning
    if ecc_0 > 0.001:  # Arbitrary tolerance
        e_vec = eccentricity_vector_hf(k, r, v)
        ref_vec = div_Vs_hf(e_vec, ecc_0)
    else:
        ref_vec = div_Vs_hf(r, norm_V_hf(r))

    h_vec = cross_VV_hf(r, v)  # Specific angular momentum vector
    h_unit = div_Vs_hf(h_vec, norm_V_hf(h_vec))
    thrust_unit = mul_Vs_hf(cross_VV_hf(h_unit, ref_vec), sign_hf(ecc_f - ecc_0))

    beta_0 = beta_hf(ecc_0, ecc_f, inc_0, inc_f, argp)

    delta_v = _delta_V_hf(circular_velocity_hf(k, a), ecc_0, ecc_f, beta_0)
    t_f = _delta_t_hf(delta_v, f)

    return thrust_unit, beta_0, delta_v, t_f


@gjit(
    "void(f,f,f,f,f,f,f,f[:],f[:],f,f[:],f[:],f[:],f[:])",
    "(),(),(),(),(),(),(),(n),(n),()->(n),(),(),()",
)
def _prepare_gf(
    k, a, ecc_0, ecc_f, inc_0, inc_f, argp, r, v, f, thrust_unit, beta_0, delta_v, t_f
):
    """
    Vectorized prepare
    """

    thrust_unit[:], beta_0[0], delta_v[0], t_f[0] = _prepare_hf(
        k, a, ecc_0, ecc_f, inc_0, inc_f, argp, array_to_V_hf(r), array_to_V_hf(v), f
    )


def change_ecc_inc_hb(k, a, ecc_0, ecc_f, inc_0, inc_f, argp, r, v, f):
    thrust_unit, beta_0, delta_v, t_f = _prepare_gf(  # pylint: disable=E1120,E0633
        k, a, ecc_0, ecc_f, inc_0, inc_f, argp, array(r), array(v), f
    )
    thrust_unit = tuple(thrust_unit)

    @hjit("V(f,V,V,f)", cache=False)
    def a_d_hf(t0, rr, vv, k_):
        nu = rv2coe_hf(k_, rr, vv, RV2COE_TOL)[-1]
        beta_ = beta_0 * sign_hf(
            cos(nu)
        )  # The sign of ß reverses at minor axis crossings

        w_ = mul_Vs_hf(
            cross_VV_hf(rr, vv), sign_hf(inc_f - inc_0) / norm_V_hf(cross_VV_hf(rr, vv))
        )
        accel_v = mul_Vs_hf(
            add_VV_hf(mul_Vs_hf(thrust_unit, cos(beta_)), mul_Vs_hf(w_, sin(beta_))), f
        )
        return accel_v

    return a_d_hf, delta_v, t_f
