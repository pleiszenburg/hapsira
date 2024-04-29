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

    Based on
    https://github.com/scipy/scipy/blob/4edfcaa3ce8a387450b6efce968572def71be089/scipy/integrate/_ivp/rk.py#L584

    Parameters
    ----------
    t : float
        Current time.
    t_old : float
        Previous time.
    h : float
        Step to use.
    rr_rold : tuple[float,float,float]
        Last values 0:3.
    vv_vold : tuple[float,float,float]
        Last values 3:6.
    F : tuple[tuple[float,...]...]
        Dense output coefficients.

    Returns
    -------
    rr : tuple[float,float,float]
        Computed values 0:3.
    vv : tuple[float,float,float]
        Computed values 3:6.
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
