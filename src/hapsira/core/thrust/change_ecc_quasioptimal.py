from math import asin

from numpy import array

from ..elements import circular_velocity_hf
from ..jit import array_to_V_hf, hjit, gjit
from ..math.linalg import cross_VV_hf, div_Vs_hf, mul_Vs_hf, norm_V_hf, sign_hf

__all__ = [
    "change_ecc_quasioptimal_hb",
]


@hjit("f(f,f,f)")
def _delta_V_hf(V_0, ecc_0, ecc_f):
    """
    Compute required increment of velocity.
    """

    return 2 / 3 * V_0 * abs(asin(ecc_0) - asin(ecc_f))


@hjit("Tuple([f,f])(f,f,f,f,f)")
def _extra_quantities_hf(k, a, ecc_0, ecc_f, f):
    """
    Extra quantities given by the model.
    """

    V_0 = circular_velocity_hf(k, a)
    delta_V_ = _delta_V_hf(V_0, ecc_0, ecc_f)
    t_f_ = delta_V_ / f

    return delta_V_, t_f_


@gjit("void(f,f,f,f,f,f[:],f[:])", "(),(),(),(),()->(),()")
def _extra_quantities_gf(k, a, ecc_0, ecc_f, f, delta_V_, t_f_):
    """
    Vectorized extra_quantities
    """

    delta_V_[0], t_f_[0] = _extra_quantities_hf(k, a, ecc_0, ecc_f, f)


@hjit("V(f,f,f,f,V,V,V)")
def _prepare_hf(k, a, ecc_0, ecc_f, e_vec, h_vec, r):
    """
    Scalar prepare
    """

    if ecc_0 > 0.001:  # Arbitrary tolerance
        ref_vec = div_Vs_hf(e_vec, ecc_0)
    else:
        ref_vec = div_Vs_hf(r, norm_V_hf(r))

    h_unit = div_Vs_hf(h_vec, norm_V_hf(h_vec))
    thrust_unit = mul_Vs_hf(cross_VV_hf(h_unit, ref_vec), sign_hf(ecc_f - ecc_0))

    return thrust_unit


@gjit("void(f,f,f,f,f[:],f[:],f[:],f[:])", "(),(),(),(),(n),(n),(n)->(n)")
def _prepare_gf(k, a, ecc_0, ecc_f, e_vec, h_vec, r, thrust_unit):
    """
    Vectorized prepare
    """

    thrust_unit[:] = _prepare_hf(
        k, a, ecc_0, ecc_f, array_to_V_hf(e_vec), array_to_V_hf(h_vec), array_to_V_hf(r)
    )


def change_ecc_quasioptimal_hb(k, a, ecc_0, ecc_f, e_vec, h_vec, r, f):
    # We fix the inertial direction at the beginning

    thrust_unit = _prepare_gf(  # pylint: disable=E1120,E0633
        k, a, ecc_0, array(ecc_f), array(e_vec), array(h_vec), r
    )
    thrust_unit = tuple(thrust_unit)

    @hjit("V(f,V,V,f)", cache=False)
    def a_d_hf(t0, rr, vv, k):
        accel_v = mul_Vs_hf(thrust_unit, f)
        return accel_v

    delta_V, t_f = _extra_quantities_gf(  # pylint: disable=E1120,E0633
        k, a, ecc_0, ecc_f, f
    )
    return a_d_hf, delta_V, t_f
