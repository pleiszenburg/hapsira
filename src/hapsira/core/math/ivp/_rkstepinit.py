from math import sqrt

from ...jit import hjit, DSIG
from ..linalg import add_VV_hf, div_VV_hf, mul_Vs_hf, norm_VV_hf, sub_VV_hf


__all__ = [
    "select_initial_step_hf",
]


@hjit(f"f(F({DSIG:s}),f,V,V,f,V,V,f,f,f,f)")
def select_initial_step_hf(fun, t0, rr, vv, argk, fr, fv, direction, order, rtol, atol):
    scale_r = (
        atol + abs(rr[0]) * rtol,
        atol + abs(rr[1]) * rtol,
        atol + abs(rr[2]) * rtol,
    )
    scale_v = (
        atol + abs(vv[0]) * rtol,
        atol + abs(vv[1]) * rtol,
        atol + abs(vv[2]) * rtol,
    )

    factor = 1 / sqrt(6)
    d0 = norm_VV_hf(div_VV_hf(rr, scale_r), div_VV_hf(vv, scale_v)) * factor
    d1 = norm_VV_hf(div_VV_hf(fr, scale_r), div_VV_hf(fv, scale_v)) * factor

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    yr1 = add_VV_hf(rr, mul_Vs_hf(fr, h0 * direction))
    yv1 = add_VV_hf(vv, mul_Vs_hf(fv, h0 * direction))

    fr1, fv1 = fun(
        t0 + h0 * direction,
        yr1,
        yv1,
        argk,
    )

    d2 = (
        norm_VV_hf(
            div_VV_hf(sub_VV_hf(fr1, fr), scale_r),
            div_VV_hf(sub_VV_hf(fv1, fv), scale_v),
        )
        / h0
    )

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1)
