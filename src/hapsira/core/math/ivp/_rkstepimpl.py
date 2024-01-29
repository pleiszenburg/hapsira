from math import inf

from ._const import ERROR_EXPONENT, KSIG, MAX_FACTOR, MIN_FACTOR, SAFETY
from ._rkstep import rk_step_hf
from ._rkerror import estimate_error_norm_V_hf
from ..linalg import abs_V_hf, add_Vs_hf, max_VV_hf, mul_Vs_hf, nextafter_hf
from ...jit import hjit, DSIG


__all__ = [
    "step_impl_hf",
]


@hjit(
    f"Tuple([b1,f,f,V,V,f,V,V,{KSIG:s}])"
    f"(F({DSIG:s}),f,f,V,V,V,V,f,f,f,f,f,{KSIG:s})"
)
def step_impl_hf(
    fun, argk, t, rr, vv, fr, fv, rtol, atol, direction, h_abs, t_bound, K
):
    min_step = 10 * abs(nextafter_hf(t, direction * inf) - t)

    if h_abs < min_step:
        h_abs = min_step

    step_accepted = False
    step_rejected = False

    while not step_accepted:
        if h_abs < min_step:
            return (
                False,
                0.0,
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.0,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                K,
            )

        h = h_abs * direction
        t_new = t + h

        if direction * (t_new - t_bound) > 0:
            t_new = t_bound

        h = t_new - t
        h_abs = abs(h)

        rr_new, vv_new, fr_new, fv_new, K_new = rk_step_hf(
            fun,
            t,
            rr,
            vv,
            fr,
            fv,
            h,
            argk,
        )

        scale_r = add_Vs_hf(
            mul_Vs_hf(
                max_VV_hf(
                    abs_V_hf(rr),
                    abs_V_hf(rr_new),
                ),
                rtol,
            ),
            atol,
        )
        scale_v = add_Vs_hf(
            mul_Vs_hf(
                max_VV_hf(
                    abs_V_hf(vv),
                    abs_V_hf(vv_new),
                ),
                rtol,
            ),
            atol,
        )
        error_norm = estimate_error_norm_V_hf(
            K_new,
            h,
            scale_r,
            scale_v,
        )

        if error_norm < 1:
            if error_norm == 0:
                factor = MAX_FACTOR
            else:
                factor = min(MAX_FACTOR, SAFETY * error_norm**ERROR_EXPONENT)

            if step_rejected:
                factor = min(1, factor)

            h_abs *= factor

            step_accepted = True
        else:
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm**ERROR_EXPONENT)
            step_rejected = True

    return True, h, t_new, rr_new, vv_new, h_abs, fr_new, fv_new, K_new
