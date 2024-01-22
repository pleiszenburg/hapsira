from typing import Callable, Tuple

import numpy as np
from numba import jit

from ._dop853_coefficients import A, B, C

__all__ = [
    "rk_step",
    "N_RV",
    "N_STAGES",
]

N_RV = 6
N_STAGES = 12

A = tuple(tuple(line) for line in A[:N_STAGES, :N_STAGES])
# B = B
C = tuple(C[:N_STAGES])


@jit(nopython=False)
def rk_step(
    fun: Callable,
    t: float,
    y: np.ndarray,
    f: np.ndarray,
    h: float,
    argk: float,
) -> Tuple[np.ndarray, np.ndarray]:
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
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
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

    assert y.shape == (N_RV,)
    assert f.shape == (N_RV,)
    # assert K.shape == (N_STAGES + 1, N_RV)

    # assert A.shape == (N_STAGES, N_STAGES)
    assert B.shape == (N_STAGES,)
    # assert C.shape == (N_STAGES,)

    K00 = f

    # for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
    #     dy = np.dot(K[:s].T, a[:s]) * h
    #     K[s] = fun(t + c * h, y + dy)

    # dy = np.dot(K[:1].T, A[1, :1]) * h
    dy = np.array(
        [
            (K00[0] * A[1][0]) * h,
            (K00[1] * A[1][0]) * h,
            (K00[2] * A[1][0]) * h,
            (K00[3] * A[1][0]) * h,
            (K00[4] * A[1][0]) * h,
            (K00[5] * A[1][0]) * h,
        ]
    )
    K01 = fun(t + C[1] * h, y + dy, argk)

    # dy = np.dot(K[:2].T, A[2, :2]) * h
    dy = np.array(
        [
            (K00[0] * A[2][0] + K01[0] * A[2][1]) * h,
            (K00[1] * A[2][0] + K01[1] * A[2][1]) * h,
            (K00[2] * A[2][0] + K01[2] * A[2][1]) * h,
            (K00[3] * A[2][0] + K01[3] * A[2][1]) * h,
            (K00[4] * A[2][0] + K01[4] * A[2][1]) * h,
            (K00[5] * A[2][0] + K01[5] * A[2][1]) * h,
        ]
    )
    K02 = fun(t + C[2] * h, y + dy, argk)

    # dy = np.dot(K[:3].T, A[3, :3]) * h
    dy = np.array(
        [
            (K00[0] * A[3][0] + K01[0] * A[3][1] + K02[0] * A[3][2]) * h,
            (K00[1] * A[3][0] + K01[1] * A[3][1] + K02[1] * A[3][2]) * h,
            (K00[2] * A[3][0] + K01[2] * A[3][1] + K02[2] * A[3][2]) * h,
            (K00[3] * A[3][0] + K01[3] * A[3][1] + K02[3] * A[3][2]) * h,
            (K00[4] * A[3][0] + K01[4] * A[3][1] + K02[4] * A[3][2]) * h,
            (K00[5] * A[3][0] + K01[5] * A[3][1] + K02[5] * A[3][2]) * h,
        ]
    )

    K03 = fun(t + C[3] * h, y + dy, argk)

    # dy = np.dot(K[:4].T, A[4, :4]) * h
    dy = np.array(
        [
            (K00[0] * A[4][0] + K01[0] * A[4][1] + K02[0] * A[4][2] + K03[0] * A[4][3])
            * h,
            (K00[1] * A[4][0] + K01[1] * A[4][1] + K02[1] * A[4][2] + K03[1] * A[4][3])
            * h,
            (K00[2] * A[4][0] + K01[2] * A[4][1] + K02[2] * A[4][2] + K03[2] * A[4][3])
            * h,
            (K00[3] * A[4][0] + K01[3] * A[4][1] + K02[3] * A[4][2] + K03[3] * A[4][3])
            * h,
            (K00[4] * A[4][0] + K01[4] * A[4][1] + K02[4] * A[4][2] + K03[4] * A[4][3])
            * h,
            (K00[5] * A[4][0] + K01[5] * A[4][1] + K02[5] * A[4][2] + K03[5] * A[4][3])
            * h,
        ]
    )
    K04 = fun(t + C[4] * h, y + dy, argk)

    # dy = np.dot(K[:5].T, A[5, :5]) * h
    dy = np.array(
        [
            (
                K00[0] * A[5][0]
                + K01[0] * A[5][1]
                + K02[0] * A[5][2]
                + K03[0] * A[5][3]
                + K04[0] * A[5][4]
            )
            * h,
            (
                K00[1] * A[5][0]
                + K01[1] * A[5][1]
                + K02[1] * A[5][2]
                + K03[1] * A[5][3]
                + K04[1] * A[5][4]
            )
            * h,
            (
                K00[2] * A[5][0]
                + K01[2] * A[5][1]
                + K02[2] * A[5][2]
                + K03[2] * A[5][3]
                + K04[2] * A[5][4]
            )
            * h,
            (
                K00[3] * A[5][0]
                + K01[3] * A[5][1]
                + K02[3] * A[5][2]
                + K03[3] * A[5][3]
                + K04[3] * A[5][4]
            )
            * h,
            (
                K00[4] * A[5][0]
                + K01[4] * A[5][1]
                + K02[4] * A[5][2]
                + K03[4] * A[5][3]
                + K04[4] * A[5][4]
            )
            * h,
            (
                K00[5] * A[5][0]
                + K01[5] * A[5][1]
                + K02[5] * A[5][2]
                + K03[5] * A[5][3]
                + K04[5] * A[5][4]
            )
            * h,
        ]
    )
    K05 = fun(t + C[5] * h, y + dy, argk)

    # dy = np.dot(K[:6].T, A[6, :6]) * h
    dy = np.array(
        [
            (
                K00[0] * A[6][0]
                + K01[0] * A[6][1]
                + K02[0] * A[6][2]
                + K03[0] * A[6][3]
                + K04[0] * A[6][4]
                + K05[0] * A[6][5]
            )
            * h,
            (
                K00[1] * A[6][0]
                + K01[1] * A[6][1]
                + K02[1] * A[6][2]
                + K03[1] * A[6][3]
                + K04[1] * A[6][4]
                + K05[1] * A[6][5]
            )
            * h,
            (
                K00[2] * A[6][0]
                + K01[2] * A[6][1]
                + K02[2] * A[6][2]
                + K03[2] * A[6][3]
                + K04[2] * A[6][4]
                + K05[2] * A[6][5]
            )
            * h,
            (
                K00[3] * A[6][0]
                + K01[3] * A[6][1]
                + K02[3] * A[6][2]
                + K03[3] * A[6][3]
                + K04[3] * A[6][4]
                + K05[3] * A[6][5]
            )
            * h,
            (
                K00[4] * A[6][0]
                + K01[4] * A[6][1]
                + K02[4] * A[6][2]
                + K03[4] * A[6][3]
                + K04[4] * A[6][4]
                + K05[4] * A[6][5]
            )
            * h,
            (
                K00[5] * A[6][0]
                + K01[5] * A[6][1]
                + K02[5] * A[6][2]
                + K03[5] * A[6][3]
                + K04[5] * A[6][4]
                + K05[5] * A[6][5]
            )
            * h,
        ]
    )
    K06 = fun(t + C[6] * h, y + dy, argk)

    # dy = np.dot(K[:7].T, A[7, :7]) * h
    dy = np.array(
        [
            (
                K00[0] * A[7][0]
                + K01[0] * A[7][1]
                + K02[0] * A[7][2]
                + K03[0] * A[7][3]
                + K04[0] * A[7][4]
                + K05[0] * A[7][5]
                + K06[0] * A[7][6]
            )
            * h,
            (
                K00[1] * A[7][0]
                + K01[1] * A[7][1]
                + K02[1] * A[7][2]
                + K03[1] * A[7][3]
                + K04[1] * A[7][4]
                + K05[1] * A[7][5]
                + K06[1] * A[7][6]
            )
            * h,
            (
                K00[2] * A[7][0]
                + K01[2] * A[7][1]
                + K02[2] * A[7][2]
                + K03[2] * A[7][3]
                + K04[2] * A[7][4]
                + K05[2] * A[7][5]
                + K06[2] * A[7][6]
            )
            * h,
            (
                K00[3] * A[7][0]
                + K01[3] * A[7][1]
                + K02[3] * A[7][2]
                + K03[3] * A[7][3]
                + K04[3] * A[7][4]
                + K05[3] * A[7][5]
                + K06[3] * A[7][6]
            )
            * h,
            (
                K00[4] * A[7][0]
                + K01[4] * A[7][1]
                + K02[4] * A[7][2]
                + K03[4] * A[7][3]
                + K04[4] * A[7][4]
                + K05[4] * A[7][5]
                + K06[4] * A[7][6]
            )
            * h,
            (
                K00[5] * A[7][0]
                + K01[5] * A[7][1]
                + K02[5] * A[7][2]
                + K03[5] * A[7][3]
                + K04[5] * A[7][4]
                + K05[5] * A[7][5]
                + K06[5] * A[7][6]
            )
            * h,
        ]
    )

    K07 = fun(t + C[7] * h, y + dy, argk)

    # dy = np.dot(K[:8].T, A[8, :8]) * h
    dy = np.array(
        [
            (
                K00[0] * A[8][0]
                + K01[0] * A[8][1]
                + K02[0] * A[8][2]
                + K03[0] * A[8][3]
                + K04[0] * A[8][4]
                + K05[0] * A[8][5]
                + K06[0] * A[8][6]
                + K07[0] * A[8][7]
            )
            * h,
            (
                K00[1] * A[8][0]
                + K01[1] * A[8][1]
                + K02[1] * A[8][2]
                + K03[1] * A[8][3]
                + K04[1] * A[8][4]
                + K05[1] * A[8][5]
                + K06[1] * A[8][6]
                + K07[1] * A[8][7]
            )
            * h,
            (
                K00[2] * A[8][0]
                + K01[2] * A[8][1]
                + K02[2] * A[8][2]
                + K03[2] * A[8][3]
                + K04[2] * A[8][4]
                + K05[2] * A[8][5]
                + K06[2] * A[8][6]
                + K07[2] * A[8][7]
            )
            * h,
            (
                K00[3] * A[8][0]
                + K01[3] * A[8][1]
                + K02[3] * A[8][2]
                + K03[3] * A[8][3]
                + K04[3] * A[8][4]
                + K05[3] * A[8][5]
                + K06[3] * A[8][6]
                + K07[3] * A[8][7]
            )
            * h,
            (
                K00[4] * A[8][0]
                + K01[4] * A[8][1]
                + K02[4] * A[8][2]
                + K03[4] * A[8][3]
                + K04[4] * A[8][4]
                + K05[4] * A[8][5]
                + K06[4] * A[8][6]
                + K07[4] * A[8][7]
            )
            * h,
            (
                K00[5] * A[8][0]
                + K01[5] * A[8][1]
                + K02[5] * A[8][2]
                + K03[5] * A[8][3]
                + K04[5] * A[8][4]
                + K05[5] * A[8][5]
                + K06[5] * A[8][6]
                + K07[5] * A[8][7]
            )
            * h,
        ]
    )
    K08 = fun(t + C[8] * h, y + dy, argk)

    # dy = np.dot(K[:9].T, A[9, :9]) * h
    dy = np.array(
        [
            (
                K00[0] * A[9][0]
                + K01[0] * A[9][1]
                + K02[0] * A[9][2]
                + K03[0] * A[9][3]
                + K04[0] * A[9][4]
                + K05[0] * A[9][5]
                + K06[0] * A[9][6]
                + K07[0] * A[9][7]
                + K08[0] * A[9][8]
            )
            * h,
            (
                K00[1] * A[9][0]
                + K01[1] * A[9][1]
                + K02[1] * A[9][2]
                + K03[1] * A[9][3]
                + K04[1] * A[9][4]
                + K05[1] * A[9][5]
                + K06[1] * A[9][6]
                + K07[1] * A[9][7]
                + K08[1] * A[9][8]
            )
            * h,
            (
                K00[2] * A[9][0]
                + K01[2] * A[9][1]
                + K02[2] * A[9][2]
                + K03[2] * A[9][3]
                + K04[2] * A[9][4]
                + K05[2] * A[9][5]
                + K06[2] * A[9][6]
                + K07[2] * A[9][7]
                + K08[2] * A[9][8]
            )
            * h,
            (
                K00[3] * A[9][0]
                + K01[3] * A[9][1]
                + K02[3] * A[9][2]
                + K03[3] * A[9][3]
                + K04[3] * A[9][4]
                + K05[3] * A[9][5]
                + K06[3] * A[9][6]
                + K07[3] * A[9][7]
                + K08[3] * A[9][8]
            )
            * h,
            (
                K00[4] * A[9][0]
                + K01[4] * A[9][1]
                + K02[4] * A[9][2]
                + K03[4] * A[9][3]
                + K04[4] * A[9][4]
                + K05[4] * A[9][5]
                + K06[4] * A[9][6]
                + K07[4] * A[9][7]
                + K08[4] * A[9][8]
            )
            * h,
            (
                K00[5] * A[9][0]
                + K01[5] * A[9][1]
                + K02[5] * A[9][2]
                + K03[5] * A[9][3]
                + K04[5] * A[9][4]
                + K05[5] * A[9][5]
                + K06[5] * A[9][6]
                + K07[5] * A[9][7]
                + K08[5] * A[9][8]
            )
            * h,
        ]
    )
    K09 = fun(t + C[9] * h, y + dy, argk)

    # dy = np.dot(K[:10].T, A[10, :10]) * h
    dy = np.array(
        [
            (
                K00[0] * A[10][0]
                + K01[0] * A[10][1]
                + K02[0] * A[10][2]
                + K03[0] * A[10][3]
                + K04[0] * A[10][4]
                + K05[0] * A[10][5]
                + K06[0] * A[10][6]
                + K07[0] * A[10][7]
                + K08[0] * A[10][8]
                + K09[0] * A[10][9]
            )
            * h,
            (
                K00[1] * A[10][0]
                + K01[1] * A[10][1]
                + K02[1] * A[10][2]
                + K03[1] * A[10][3]
                + K04[1] * A[10][4]
                + K05[1] * A[10][5]
                + K06[1] * A[10][6]
                + K07[1] * A[10][7]
                + K08[1] * A[10][8]
                + K09[1] * A[10][9]
            )
            * h,
            (
                K00[2] * A[10][0]
                + K01[2] * A[10][1]
                + K02[2] * A[10][2]
                + K03[2] * A[10][3]
                + K04[2] * A[10][4]
                + K05[2] * A[10][5]
                + K06[2] * A[10][6]
                + K07[2] * A[10][7]
                + K08[2] * A[10][8]
                + K09[2] * A[10][9]
            )
            * h,
            (
                K00[3] * A[10][0]
                + K01[3] * A[10][1]
                + K02[3] * A[10][2]
                + K03[3] * A[10][3]
                + K04[3] * A[10][4]
                + K05[3] * A[10][5]
                + K06[3] * A[10][6]
                + K07[3] * A[10][7]
                + K08[3] * A[10][8]
                + K09[3] * A[10][9]
            )
            * h,
            (
                K00[4] * A[10][0]
                + K01[4] * A[10][1]
                + K02[4] * A[10][2]
                + K03[4] * A[10][3]
                + K04[4] * A[10][4]
                + K05[4] * A[10][5]
                + K06[4] * A[10][6]
                + K07[4] * A[10][7]
                + K08[4] * A[10][8]
                + K09[4] * A[10][9]
            )
            * h,
            (
                K00[5] * A[10][0]
                + K01[5] * A[10][1]
                + K02[5] * A[10][2]
                + K03[5] * A[10][3]
                + K04[5] * A[10][4]
                + K05[5] * A[10][5]
                + K06[5] * A[10][6]
                + K07[5] * A[10][7]
                + K08[5] * A[10][8]
                + K09[5] * A[10][9]
            )
            * h,
        ]
    )
    K10 = fun(t + C[10] * h, y + dy, argk)

    # dy = np.dot(K[:11].T, A[11, :11]) * h
    dy = np.array(
        [
            (
                K00[0] * A[11][0]
                + K01[0] * A[11][1]
                + K02[0] * A[11][2]
                + K03[0] * A[11][3]
                + K04[0] * A[11][4]
                + K05[0] * A[11][5]
                + K06[0] * A[11][6]
                + K07[0] * A[11][7]
                + K08[0] * A[11][8]
                + K09[0] * A[11][9]
                + K10[0] * A[11][10]
            )
            * h,
            (
                K00[1] * A[11][0]
                + K01[1] * A[11][1]
                + K02[1] * A[11][2]
                + K03[1] * A[11][3]
                + K04[1] * A[11][4]
                + K05[1] * A[11][5]
                + K06[1] * A[11][6]
                + K07[1] * A[11][7]
                + K08[1] * A[11][8]
                + K09[1] * A[11][9]
                + K10[1] * A[11][10]
            )
            * h,
            (
                K00[2] * A[11][0]
                + K01[2] * A[11][1]
                + K02[2] * A[11][2]
                + K03[2] * A[11][3]
                + K04[2] * A[11][4]
                + K05[2] * A[11][5]
                + K06[2] * A[11][6]
                + K07[2] * A[11][7]
                + K08[2] * A[11][8]
                + K09[2] * A[11][9]
                + K10[2] * A[11][10]
            )
            * h,
            (
                K00[3] * A[11][0]
                + K01[3] * A[11][1]
                + K02[3] * A[11][2]
                + K03[3] * A[11][3]
                + K04[3] * A[11][4]
                + K05[3] * A[11][5]
                + K06[3] * A[11][6]
                + K07[3] * A[11][7]
                + K08[3] * A[11][8]
                + K09[3] * A[11][9]
                + K10[3] * A[11][10]
            )
            * h,
            (
                K00[4] * A[11][0]
                + K01[4] * A[11][1]
                + K02[4] * A[11][2]
                + K03[4] * A[11][3]
                + K04[4] * A[11][4]
                + K05[4] * A[11][5]
                + K06[4] * A[11][6]
                + K07[4] * A[11][7]
                + K08[4] * A[11][8]
                + K09[4] * A[11][9]
                + K10[4] * A[11][10]
            )
            * h,
            (
                K00[5] * A[11][0]
                + K01[5] * A[11][1]
                + K02[5] * A[11][2]
                + K03[5] * A[11][3]
                + K04[5] * A[11][4]
                + K05[5] * A[11][5]
                + K06[5] * A[11][6]
                + K07[5] * A[11][7]
                + K08[5] * A[11][8]
                + K09[5] * A[11][9]
                + K10[5] * A[11][10]
            )
            * h,
        ]
    )
    K11 = fun(t + C[11] * h, y + dy, argk)

    K_ = np.array(
        [
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
        ]
    ).T

    y_new = y + h * np.dot(K_, B)
    f_new = fun(t + h, y_new, argk)

    K12 = f_new

    assert y_new.shape == (N_RV,)
    assert f_new.shape == (N_RV,)

    return (
        y_new,
        f_new,
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
