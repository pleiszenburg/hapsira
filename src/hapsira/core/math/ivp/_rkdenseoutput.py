import numpy as np

from ._const import (
    N_RV,
    N_STAGES,
    N_STAGES_EXTENDED,
)
from ._dop853_coefficients import A as _A, C as _C, D as _D
from ...jit import array_to_V_hf
from ...math.linalg import (
    add_VV_hf,
    mul_Vs_hf,
    sub_VV_hf,
)

__all__ = [
    "dense_output_hf",
]


A_EXTRA = _A[N_STAGES + 1 :]
C_EXTRA = _C[N_STAGES + 1 :]
D = _D


# TODO compile
def dense_output_hf(
    fun, argk, t_old, t, h_previous, rr, vv, rr_old, vv_old, fr, fv, K_
):
    """Compute a local interpolant over the last successful step.

    Returns
    -------
    sol : `DenseOutput`
        Local interpolant over the last successful step.
    """

    assert t_old is not None
    assert t != t_old

    Ke = np.empty((N_STAGES_EXTENDED, N_RV), dtype=float)
    Ke[: N_STAGES + 1, :] = np.array(K_)

    h = h_previous

    for s, (a, c) in enumerate(zip(A_EXTRA, C_EXTRA), start=N_STAGES + 1):
        dy = np.dot(Ke[:s].T, a[:s]) * h
        rr_ = add_VV_hf(rr_old, array_to_V_hf(dy[:3]))
        vv_ = add_VV_hf(vv_old, array_to_V_hf(dy[3:]))
        rr_, vv_ = fun(
            t_old + c * h,
            rr_,
            vv_,
            argk,
        )  # TODO call into hf
        Ke[s] = np.array([*rr_, *vv_])

    fr_old = array_to_V_hf(Ke[0, :3])
    fv_old = array_to_V_hf(Ke[0, 3:])

    delta_rr = sub_VV_hf(rr, rr_old)
    delta_vv = sub_VV_hf(vv, vv_old)

    F00 = *delta_rr, *delta_vv
    F01 = *sub_VV_hf(mul_Vs_hf(fr_old, h), delta_rr), *sub_VV_hf(
        mul_Vs_hf(fv_old, h), delta_vv
    )
    F02 = *sub_VV_hf(
        mul_Vs_hf(delta_rr, 2), mul_Vs_hf(add_VV_hf(fr, fr_old), h)
    ), *sub_VV_hf(mul_Vs_hf(delta_vv, 2), mul_Vs_hf(add_VV_hf(fv, fv_old), h))

    F03, F04, F05, F06 = tuple(
        tuple(float(number) for number in line) for line in (h * np.dot(D, Ke))
    )  # TODO

    return (
        t_old,
        t - t_old,  # h
        rr_old,
        vv_old,
        (F00, F01, F02, F03, F04, F05, F06),
    )
