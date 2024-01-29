from ._const import FSIG
from ..linalg import add_VV_hf, mul_Vs_hf
from ...jit import hjit


__all__ = [
    "dense_interp_brentq_hb",
    "dense_interp_hf",
    "DENSE_SIG",
]


DENSE_SIG = f"f,f,V,V,{FSIG:s}"


@hjit(f"Tuple([V,V])(f,{DENSE_SIG:s})")
def dense_interp_hf(t, t_old, h, rr_old, vv_old, F):
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
    rr_new = (0.0, 0.0, 0.0)
    vv_new = (0.0, 0.0, 0.0)

    for idx, f in enumerate((F06, F05, F04, F03, F02, F01, F00)):
        rr_new = add_VV_hf(rr_new, f[:3])
        vv_new = add_VV_hf(vv_new, f[3:])

        if idx % 2 == 0:
            rr_new = mul_Vs_hf(rr_new, x)
            vv_new = mul_Vs_hf(vv_new, x)
        else:
            rr_new = mul_Vs_hf(rr_new, 1 - x)
            vv_new = mul_Vs_hf(vv_new, 1 - x)

    rr_new = add_VV_hf(rr_new, rr_old)
    vv_new = add_VV_hf(vv_new, vv_old)

    return rr_new, vv_new


def dense_interp_brentq_hb(func):
    @hjit(f"f(f,{DENSE_SIG:s},f)", cache=False)
    def event_wrapper(t, t_old, h, rr_old, vv_old, F, argk):
        rr, vv = dense_interp_hf(t, t_old, h, rr_old, vv_old, F)
        return func(t, rr, vv, argk)

    return event_wrapper
