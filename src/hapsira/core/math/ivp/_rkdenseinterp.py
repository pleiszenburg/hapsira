from ._const import DENSE_SIG
from ..linalg import add_VV_hf, mul_Vs_hf
from ...jit import hjit


__all__ = [
    "dop853_dense_interp_brentq_hb",
    "dop853_dense_interp_hf",
]


@hjit(f"Tuple([V,V])(f,{DENSE_SIG:s})")
def dop853_dense_interp_hf(t, t_old, h, rr_old, vv_old, F):
    """
    Local interpolant over step made by an ODE solver.
    Evaluate the interpolant.

    Parameters
    ----------
    t : float or array_like with shape (n_points,)
        Points to evaluate the solution at.

    Returns
    -------
    y : ndarray, shape (n,) or (n, n_points)
        Computed values. Shape depends on whether `t` was a scalar or a
        1-D array.
    """

    F00, F01, F02, F03, F04, F05, F06 = F

    x = (t - t_old) / h

    rr_new = mul_Vs_hf(F06[:3], x)
    vv_new = mul_Vs_hf(F06[3:], x)

    rr_new = add_VV_hf(rr_new, F05[:3])
    vv_new = add_VV_hf(vv_new, F05[3:])
    rr_new = mul_Vs_hf(rr_new, 1 - x)
    vv_new = mul_Vs_hf(vv_new, 1 - x)

    rr_new = add_VV_hf(rr_new, F04[:3])
    vv_new = add_VV_hf(vv_new, F04[3:])
    rr_new = mul_Vs_hf(rr_new, x)
    vv_new = mul_Vs_hf(vv_new, x)

    rr_new = add_VV_hf(rr_new, F03[:3])
    vv_new = add_VV_hf(vv_new, F03[3:])
    rr_new = mul_Vs_hf(rr_new, 1 - x)
    vv_new = mul_Vs_hf(vv_new, 1 - x)

    rr_new = add_VV_hf(rr_new, F02[:3])
    vv_new = add_VV_hf(vv_new, F02[3:])
    rr_new = mul_Vs_hf(rr_new, x)
    vv_new = mul_Vs_hf(vv_new, x)

    rr_new = add_VV_hf(rr_new, F01[:3])
    vv_new = add_VV_hf(vv_new, F01[3:])
    rr_new = mul_Vs_hf(rr_new, 1 - x)
    vv_new = mul_Vs_hf(vv_new, 1 - x)

    rr_new = add_VV_hf(rr_new, F00[:3])
    vv_new = add_VV_hf(vv_new, F00[3:])
    rr_new = mul_Vs_hf(rr_new, x)
    vv_new = mul_Vs_hf(vv_new, x)

    rr_new = add_VV_hf(rr_new, rr_old)
    vv_new = add_VV_hf(vv_new, vv_old)

    return rr_new, vv_new


def dop853_dense_interp_brentq_hb(func):
    @hjit(f"f(f,{DENSE_SIG:s},f)", cache=False)
    def event_wrapper(t, t_old, h, rr_old, vv_old, F, argk):
        rr, vv = dop853_dense_interp_hf(t, t_old, h, rr_old, vv_old, F)
        return func(t, rr, vv, argk)

    return event_wrapper
