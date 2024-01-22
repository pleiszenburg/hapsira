from typing import Callable, Tuple

from numpy import float32 as f4

from ._dop853_coefficients import A as _A, B as _B, C as _C
from ..linalg import add_VV_hf
from ...jit import hjit, DSIG

__all__ = [
    "rk_step_hf",
    "N_RV",
    "N_STAGES",
]

N_RV = 6
N_STAGES = 12

A01 = tuple(f4(number) for number in _A[1, :N_STAGES])
A02 = tuple(f4(number) for number in _A[2, :N_STAGES])
A03 = tuple(f4(number) for number in _A[3, :N_STAGES])
A04 = tuple(f4(number) for number in _A[4, :N_STAGES])
A05 = tuple(f4(number) for number in _A[5, :N_STAGES])
A06 = tuple(f4(number) for number in _A[6, :N_STAGES])
A07 = tuple(f4(number) for number in _A[7, :N_STAGES])
A08 = tuple(f4(number) for number in _A[8, :N_STAGES])
A09 = tuple(f4(number) for number in _A[9, :N_STAGES])
A10 = tuple(f4(number) for number in _A[10, :N_STAGES])
A11 = tuple(f4(number) for number in _A[11, :N_STAGES])
B = tuple(f4(number) for number in _B)
C = tuple(f4(number) for number in _C[:N_STAGES])

_KSIG = (
    "Tuple(["
    + ",".join(["Tuple([" + ",".join(["f"] * N_RV) + "])"] * (N_STAGES + 1))
    + "])"
)


@hjit(f"Tuple([V,V,V,V,{_KSIG:s}])(F({DSIG:s}),f,V,V,V,V,f,f)")
def rk_step_hf(
    fun: Callable,
    t: float,
    rr: tuple[float, float, float],
    vv: tuple[float, float, float],
    fr: tuple[float, float, float],
    fv: tuple[float, float, float],
    h: float,
    argk: float,
) -> Tuple[Tuple, Tuple, Tuple, Tuple, Tuple]:
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    r : tuple[float,float,float]
        Current r.
    v : tuple[float,float,float]
        Current v.
    fr : tuple[float,float,float]
        Current value of the derivative, i.e., ``fun(x, y)``.
    fv : tuple[float,float,float]
        Current value of the derivative, i.e., ``fun(x, y)``.
    h : float
        Step to use.

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients

    Const
    -----
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """

    K00 = *fr, *fv

    dr = (
        (K00[0] * A01[0]) * h,
        (K00[1] * A01[0]) * h,
        (K00[2] * A01[0]) * h,
    )
    dv = (
        (K00[3] * A01[0]) * h,
        (K00[4] * A01[0]) * h,
        (K00[5] * A01[0]) * h,
    )
    fr, fv = fun(
        t + C[1] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K01 = *fr, *fv

    dr = (
        (K00[0] * A02[0] + K01[0] * A02[1]) * h,
        (K00[1] * A02[0] + K01[1] * A02[1]) * h,
        (K00[2] * A02[0] + K01[2] * A02[1]) * h,
    )
    dv = (
        (K00[3] * A02[0] + K01[3] * A02[1]) * h,
        (K00[4] * A02[0] + K01[4] * A02[1]) * h,
        (K00[5] * A02[0] + K01[5] * A02[1]) * h,
    )
    fr, fv = fun(
        t + C[2] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K02 = *fr, *fv

    dr = (
        (K00[0] * A03[0] + K01[0] * A03[1] + K02[0] * A03[2]) * h,
        (K00[1] * A03[0] + K01[1] * A03[1] + K02[1] * A03[2]) * h,
        (K00[2] * A03[0] + K01[2] * A03[1] + K02[2] * A03[2]) * h,
    )
    dv = (
        (K00[3] * A03[0] + K01[3] * A03[1] + K02[3] * A03[2]) * h,
        (K00[4] * A03[0] + K01[4] * A03[1] + K02[4] * A03[2]) * h,
        (K00[5] * A03[0] + K01[5] * A03[1] + K02[5] * A03[2]) * h,
    )
    fr, fv = fun(
        t + C[3] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K03 = *fr, *fv

    dr = (
        (K00[0] * A04[0] + K01[0] * A04[1] + K02[0] * A04[2] + K03[0] * A04[3]) * h,
        (K00[1] * A04[0] + K01[1] * A04[1] + K02[1] * A04[2] + K03[1] * A04[3]) * h,
        (K00[2] * A04[0] + K01[2] * A04[1] + K02[2] * A04[2] + K03[2] * A04[3]) * h,
    )
    dv = (
        (K00[3] * A04[0] + K01[3] * A04[1] + K02[3] * A04[2] + K03[3] * A04[3]) * h,
        (K00[4] * A04[0] + K01[4] * A04[1] + K02[4] * A04[2] + K03[4] * A04[3]) * h,
        (K00[5] * A04[0] + K01[5] * A04[1] + K02[5] * A04[2] + K03[5] * A04[3]) * h,
    )
    fr, fv = fun(
        t + C[4] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K04 = *fr, *fv

    dr = (
        (
            K00[0] * A05[0]
            + K01[0] * A05[1]
            + K02[0] * A05[2]
            + K03[0] * A05[3]
            + K04[0] * A05[4]
        )
        * h,
        (
            K00[1] * A05[0]
            + K01[1] * A05[1]
            + K02[1] * A05[2]
            + K03[1] * A05[3]
            + K04[1] * A05[4]
        )
        * h,
        (
            K00[2] * A05[0]
            + K01[2] * A05[1]
            + K02[2] * A05[2]
            + K03[2] * A05[3]
            + K04[2] * A05[4]
        )
        * h,
    )
    dv = (
        (
            K00[3] * A05[0]
            + K01[3] * A05[1]
            + K02[3] * A05[2]
            + K03[3] * A05[3]
            + K04[3] * A05[4]
        )
        * h,
        (
            K00[4] * A05[0]
            + K01[4] * A05[1]
            + K02[4] * A05[2]
            + K03[4] * A05[3]
            + K04[4] * A05[4]
        )
        * h,
        (
            K00[5] * A05[0]
            + K01[5] * A05[1]
            + K02[5] * A05[2]
            + K03[5] * A05[3]
            + K04[5] * A05[4]
        )
        * h,
    )
    fr, fv = fun(
        t + C[5] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K05 = *fr, *fv

    dr = (
        (
            K00[0] * A06[0]
            + K01[0] * A06[1]
            + K02[0] * A06[2]
            + K03[0] * A06[3]
            + K04[0] * A06[4]
            + K05[0] * A06[5]
        )
        * h,
        (
            K00[1] * A06[0]
            + K01[1] * A06[1]
            + K02[1] * A06[2]
            + K03[1] * A06[3]
            + K04[1] * A06[4]
            + K05[1] * A06[5]
        )
        * h,
        (
            K00[2] * A06[0]
            + K01[2] * A06[1]
            + K02[2] * A06[2]
            + K03[2] * A06[3]
            + K04[2] * A06[4]
            + K05[2] * A06[5]
        )
        * h,
    )
    dv = (
        (
            K00[3] * A06[0]
            + K01[3] * A06[1]
            + K02[3] * A06[2]
            + K03[3] * A06[3]
            + K04[3] * A06[4]
            + K05[3] * A06[5]
        )
        * h,
        (
            K00[4] * A06[0]
            + K01[4] * A06[1]
            + K02[4] * A06[2]
            + K03[4] * A06[3]
            + K04[4] * A06[4]
            + K05[4] * A06[5]
        )
        * h,
        (
            K00[5] * A06[0]
            + K01[5] * A06[1]
            + K02[5] * A06[2]
            + K03[5] * A06[3]
            + K04[5] * A06[4]
            + K05[5] * A06[5]
        )
        * h,
    )
    fr, fv = fun(
        t + C[6] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K06 = *fr, *fv

    dr = (
        (
            K00[0] * A07[0]
            + K01[0] * A07[1]
            + K02[0] * A07[2]
            + K03[0] * A07[3]
            + K04[0] * A07[4]
            + K05[0] * A07[5]
            + K06[0] * A07[6]
        )
        * h,
        (
            K00[1] * A07[0]
            + K01[1] * A07[1]
            + K02[1] * A07[2]
            + K03[1] * A07[3]
            + K04[1] * A07[4]
            + K05[1] * A07[5]
            + K06[1] * A07[6]
        )
        * h,
        (
            K00[2] * A07[0]
            + K01[2] * A07[1]
            + K02[2] * A07[2]
            + K03[2] * A07[3]
            + K04[2] * A07[4]
            + K05[2] * A07[5]
            + K06[2] * A07[6]
        )
        * h,
    )
    dv = (
        (
            K00[3] * A07[0]
            + K01[3] * A07[1]
            + K02[3] * A07[2]
            + K03[3] * A07[3]
            + K04[3] * A07[4]
            + K05[3] * A07[5]
            + K06[3] * A07[6]
        )
        * h,
        (
            K00[4] * A07[0]
            + K01[4] * A07[1]
            + K02[4] * A07[2]
            + K03[4] * A07[3]
            + K04[4] * A07[4]
            + K05[4] * A07[5]
            + K06[4] * A07[6]
        )
        * h,
        (
            K00[5] * A07[0]
            + K01[5] * A07[1]
            + K02[5] * A07[2]
            + K03[5] * A07[3]
            + K04[5] * A07[4]
            + K05[5] * A07[5]
            + K06[5] * A07[6]
        )
        * h,
    )
    fr, fv = fun(
        t + C[7] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K07 = *fr, *fv

    dr = (
        (
            K00[0] * A08[0]
            + K01[0] * A08[1]
            + K02[0] * A08[2]
            + K03[0] * A08[3]
            + K04[0] * A08[4]
            + K05[0] * A08[5]
            + K06[0] * A08[6]
            + K07[0] * A08[7]
        )
        * h,
        (
            K00[1] * A08[0]
            + K01[1] * A08[1]
            + K02[1] * A08[2]
            + K03[1] * A08[3]
            + K04[1] * A08[4]
            + K05[1] * A08[5]
            + K06[1] * A08[6]
            + K07[1] * A08[7]
        )
        * h,
        (
            K00[2] * A08[0]
            + K01[2] * A08[1]
            + K02[2] * A08[2]
            + K03[2] * A08[3]
            + K04[2] * A08[4]
            + K05[2] * A08[5]
            + K06[2] * A08[6]
            + K07[2] * A08[7]
        )
        * h,
    )
    dv = (
        (
            K00[3] * A08[0]
            + K01[3] * A08[1]
            + K02[3] * A08[2]
            + K03[3] * A08[3]
            + K04[3] * A08[4]
            + K05[3] * A08[5]
            + K06[3] * A08[6]
            + K07[3] * A08[7]
        )
        * h,
        (
            K00[4] * A08[0]
            + K01[4] * A08[1]
            + K02[4] * A08[2]
            + K03[4] * A08[3]
            + K04[4] * A08[4]
            + K05[4] * A08[5]
            + K06[4] * A08[6]
            + K07[4] * A08[7]
        )
        * h,
        (
            K00[5] * A08[0]
            + K01[5] * A08[1]
            + K02[5] * A08[2]
            + K03[5] * A08[3]
            + K04[5] * A08[4]
            + K05[5] * A08[5]
            + K06[5] * A08[6]
            + K07[5] * A08[7]
        )
        * h,
    )
    fr, fv = fun(
        t + C[8] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K08 = *fr, *fv

    dr = (
        (
            K00[0] * A09[0]
            + K01[0] * A09[1]
            + K02[0] * A09[2]
            + K03[0] * A09[3]
            + K04[0] * A09[4]
            + K05[0] * A09[5]
            + K06[0] * A09[6]
            + K07[0] * A09[7]
            + K08[0] * A09[8]
        )
        * h,
        (
            K00[1] * A09[0]
            + K01[1] * A09[1]
            + K02[1] * A09[2]
            + K03[1] * A09[3]
            + K04[1] * A09[4]
            + K05[1] * A09[5]
            + K06[1] * A09[6]
            + K07[1] * A09[7]
            + K08[1] * A09[8]
        )
        * h,
        (
            K00[2] * A09[0]
            + K01[2] * A09[1]
            + K02[2] * A09[2]
            + K03[2] * A09[3]
            + K04[2] * A09[4]
            + K05[2] * A09[5]
            + K06[2] * A09[6]
            + K07[2] * A09[7]
            + K08[2] * A09[8]
        )
        * h,
    )
    dv = (
        (
            K00[3] * A09[0]
            + K01[3] * A09[1]
            + K02[3] * A09[2]
            + K03[3] * A09[3]
            + K04[3] * A09[4]
            + K05[3] * A09[5]
            + K06[3] * A09[6]
            + K07[3] * A09[7]
            + K08[3] * A09[8]
        )
        * h,
        (
            K00[4] * A09[0]
            + K01[4] * A09[1]
            + K02[4] * A09[2]
            + K03[4] * A09[3]
            + K04[4] * A09[4]
            + K05[4] * A09[5]
            + K06[4] * A09[6]
            + K07[4] * A09[7]
            + K08[4] * A09[8]
        )
        * h,
        (
            K00[5] * A09[0]
            + K01[5] * A09[1]
            + K02[5] * A09[2]
            + K03[5] * A09[3]
            + K04[5] * A09[4]
            + K05[5] * A09[5]
            + K06[5] * A09[6]
            + K07[5] * A09[7]
            + K08[5] * A09[8]
        )
        * h,
    )
    fr, fv = fun(
        t + C[9] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K09 = *fr, *fv

    dr = (
        (
            K00[0] * A10[0]
            + K01[0] * A10[1]
            + K02[0] * A10[2]
            + K03[0] * A10[3]
            + K04[0] * A10[4]
            + K05[0] * A10[5]
            + K06[0] * A10[6]
            + K07[0] * A10[7]
            + K08[0] * A10[8]
            + K09[0] * A10[9]
        )
        * h,
        (
            K00[1] * A10[0]
            + K01[1] * A10[1]
            + K02[1] * A10[2]
            + K03[1] * A10[3]
            + K04[1] * A10[4]
            + K05[1] * A10[5]
            + K06[1] * A10[6]
            + K07[1] * A10[7]
            + K08[1] * A10[8]
            + K09[1] * A10[9]
        )
        * h,
        (
            K00[2] * A10[0]
            + K01[2] * A10[1]
            + K02[2] * A10[2]
            + K03[2] * A10[3]
            + K04[2] * A10[4]
            + K05[2] * A10[5]
            + K06[2] * A10[6]
            + K07[2] * A10[7]
            + K08[2] * A10[8]
            + K09[2] * A10[9]
        )
        * h,
    )
    dv = (
        (
            K00[3] * A10[0]
            + K01[3] * A10[1]
            + K02[3] * A10[2]
            + K03[3] * A10[3]
            + K04[3] * A10[4]
            + K05[3] * A10[5]
            + K06[3] * A10[6]
            + K07[3] * A10[7]
            + K08[3] * A10[8]
            + K09[3] * A10[9]
        )
        * h,
        (
            K00[4] * A10[0]
            + K01[4] * A10[1]
            + K02[4] * A10[2]
            + K03[4] * A10[3]
            + K04[4] * A10[4]
            + K05[4] * A10[5]
            + K06[4] * A10[6]
            + K07[4] * A10[7]
            + K08[4] * A10[8]
            + K09[4] * A10[9]
        )
        * h,
        (
            K00[5] * A10[0]
            + K01[5] * A10[1]
            + K02[5] * A10[2]
            + K03[5] * A10[3]
            + K04[5] * A10[4]
            + K05[5] * A10[5]
            + K06[5] * A10[6]
            + K07[5] * A10[7]
            + K08[5] * A10[8]
            + K09[5] * A10[9]
        )
        * h,
    )
    fr, fv = fun(
        t + C[10] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K10 = *fr, *fv

    dr = (
        (
            K00[0] * A11[0]
            + K01[0] * A11[1]
            + K02[0] * A11[2]
            + K03[0] * A11[3]
            + K04[0] * A11[4]
            + K05[0] * A11[5]
            + K06[0] * A11[6]
            + K07[0] * A11[7]
            + K08[0] * A11[8]
            + K09[0] * A11[9]
            + K10[0] * A11[10]
        )
        * h,
        (
            K00[1] * A11[0]
            + K01[1] * A11[1]
            + K02[1] * A11[2]
            + K03[1] * A11[3]
            + K04[1] * A11[4]
            + K05[1] * A11[5]
            + K06[1] * A11[6]
            + K07[1] * A11[7]
            + K08[1] * A11[8]
            + K09[1] * A11[9]
            + K10[1] * A11[10]
        )
        * h,
        (
            K00[2] * A11[0]
            + K01[2] * A11[1]
            + K02[2] * A11[2]
            + K03[2] * A11[3]
            + K04[2] * A11[4]
            + K05[2] * A11[5]
            + K06[2] * A11[6]
            + K07[2] * A11[7]
            + K08[2] * A11[8]
            + K09[2] * A11[9]
            + K10[2] * A11[10]
        )
        * h,
    )
    dv = (
        (
            K00[3] * A11[0]
            + K01[3] * A11[1]
            + K02[3] * A11[2]
            + K03[3] * A11[3]
            + K04[3] * A11[4]
            + K05[3] * A11[5]
            + K06[3] * A11[6]
            + K07[3] * A11[7]
            + K08[3] * A11[8]
            + K09[3] * A11[9]
            + K10[3] * A11[10]
        )
        * h,
        (
            K00[4] * A11[0]
            + K01[4] * A11[1]
            + K02[4] * A11[2]
            + K03[4] * A11[3]
            + K04[4] * A11[4]
            + K05[4] * A11[5]
            + K06[4] * A11[6]
            + K07[4] * A11[7]
            + K08[4] * A11[8]
            + K09[4] * A11[9]
            + K10[4] * A11[10]
        )
        * h,
        (
            K00[5] * A11[0]
            + K01[5] * A11[1]
            + K02[5] * A11[2]
            + K03[5] * A11[3]
            + K04[5] * A11[4]
            + K05[5] * A11[5]
            + K06[5] * A11[6]
            + K07[5] * A11[7]
            + K08[5] * A11[8]
            + K09[5] * A11[9]
            + K10[5] * A11[10]
        )
        * h,
    )
    fr, fv = fun(
        t + C[11] * h,
        add_VV_hf(rr, dr),
        add_VV_hf(vv, dv),
        argk,
    )
    K11 = *fr, *fv

    dr = (
        (
            K00[0] * B[0]
            + K01[0] * B[1]
            + K02[0] * B[2]
            + K03[0] * B[3]
            + K04[0] * B[4]
            + K05[0] * B[5]
            + K06[0] * B[6]
            + K07[0] * B[7]
            + K08[0] * B[8]
            + K09[0] * B[9]
            + K10[0] * B[10]
            + K11[0] * B[11]
        )
        * h,
        (
            K00[1] * B[0]
            + K01[1] * B[1]
            + K02[1] * B[2]
            + K03[1] * B[3]
            + K04[1] * B[4]
            + K05[1] * B[5]
            + K06[1] * B[6]
            + K07[1] * B[7]
            + K08[1] * B[8]
            + K09[1] * B[9]
            + K10[1] * B[10]
            + K11[1] * B[11]
        )
        * h,
        (
            K00[2] * B[0]
            + K01[2] * B[1]
            + K02[2] * B[2]
            + K03[2] * B[3]
            + K04[2] * B[4]
            + K05[2] * B[5]
            + K06[2] * B[6]
            + K07[2] * B[7]
            + K08[2] * B[8]
            + K09[2] * B[9]
            + K10[2] * B[10]
            + K11[2] * B[11]
        )
        * h,
    )
    dv = (
        (
            K00[3] * B[0]
            + K01[3] * B[1]
            + K02[3] * B[2]
            + K03[3] * B[3]
            + K04[3] * B[4]
            + K05[3] * B[5]
            + K06[3] * B[6]
            + K07[3] * B[7]
            + K08[3] * B[8]
            + K09[3] * B[9]
            + K10[3] * B[10]
            + K11[3] * B[11]
        )
        * h,
        (
            K00[4] * B[0]
            + K01[4] * B[1]
            + K02[4] * B[2]
            + K03[4] * B[3]
            + K04[4] * B[4]
            + K05[4] * B[5]
            + K06[4] * B[6]
            + K07[4] * B[7]
            + K08[4] * B[8]
            + K09[4] * B[9]
            + K10[4] * B[10]
            + K11[4] * B[11]
        )
        * h,
        (
            K00[5] * B[0]
            + K01[4] * B[1]
            + K02[5] * B[2]
            + K03[5] * B[3]
            + K04[5] * B[4]
            + K05[5] * B[5]
            + K06[5] * B[6]
            + K07[5] * B[7]
            + K08[5] * B[8]
            + K09[5] * B[9]
            + K10[5] * B[10]
            + K11[5] * B[11]
        )
        * h,
    )
    rr_new = add_VV_hf(rr, dr)
    vv_new = add_VV_hf(vv, dr)
    fr_new, fv_new = fun(t + h, rr_new, vv_new, argk)
    K12 = *fr_new, *fv_new

    return (
        rr_new,
        vv_new,
        fr_new,
        fv_new,
        (
            K00,
            K01,
            K02,
            K03,
            K04,
            K05,
            K06,
            K07,
            K08,
            K09,
            K10,
            K11,
            K12,
        ),
    )
