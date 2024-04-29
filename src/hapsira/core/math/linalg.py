from math import fabs, inf, sqrt

from ..jit import hjit, vjit

__all__ = [
    "abs_V_hf",
    "add_Vs_hf",
    "add_VV_hf",
    "cross_VV_hf",
    "div_Vs_hf",
    "div_VV_hf",
    "div_ss_hf",
    "matmul_MM_hf",
    "matmul_MV_hf",
    "matmul_VM_hf",
    "matmul_VV_hf",
    "max_VV_hf",
    "mul_Vs_hf",
    "mul_VV_hf",
    "norm_V_hf",
    "norm_V_vf",
    "norm_VV_hf",
    "sign_hf",
    "sub_VV_hf",
    "transpose_M_hf",
]


@hjit("V(V)", inline=True)
def abs_V_hf(a):
    """
    Abs 3D vector of 3D vector element-wise.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector

    Returns
    -------
    b : tuple[float,float,float]
        Vector

    """

    return fabs(a[0]), fabs(a[1]), fabs(a[2])


@hjit("V(V,f)", inline=True)
def add_Vs_hf(a, b):
    """
    Adds a 3D vector and a scalar element-wise.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : float
        Scalar

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return a[0] + b, a[1] + b, a[2] + b


@hjit("V(V,V)", inline=True)
def add_VV_hf(a, b):
    """
    Adds two 3D vectors.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : tuple[float,float,float]
        Vector

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


@hjit("V(V,V)", inline=True)
def cross_VV_hf(a, b):
    """
    Cross-product of two 3D vectors.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : tuple[float,float,float]
        Vector

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@hjit("f(f,f)", inline=True)
def div_ss_hf(a, b):
    """
    Division of two scalars. Similar to `numpy.divide` as it returns
    +/- (depending on the sign of `a`) infinity if `b` is zero.
    Required for compatibility if `core` is not compiled for debugging purposes.
    Inline-compiled by default.

    Parameters
    ----------
    a : float
        Scalar
    b : float
        Scalar

    Returns
    -------
    c : float
        Scalar

    """

    if b == 0:
        return inf if a >= 0 else -inf
    return a / b


@hjit("V(V,V)", inline=True)
def div_VV_hf(a, b):
    """
    Division of two 3D vectors element-wise. Similar to `numpy.divide` as
    it returns +/- (depending on the sign of `a`) infinity if `b` is zero.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : tuple[float,float,float]
        Vector

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return div_ss_hf(a[0], b[0]), div_ss_hf(a[1], b[1]), div_ss_hf(a[2], b[2])


@hjit("V(V,f)", inline=True)
def div_Vs_hf(a, b):
    """
    Division of a 3D vector by a scalar element-wise. Similar to `numpy.divide` as
    it returns +/- (depending on the sign of `a`) infinity if `b` is zero.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : float
        Scalar

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return div_ss_hf(a[0], b), div_ss_hf(a[1], b), div_ss_hf(a[2], b)


@hjit("M(M,M)", inline=True)
def matmul_MM_hf(a, b):
    """
    Matmul (dot product) between two 3x3 matrices.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[tuple[float,float,float],tuple[float,float,float],tuple[float,float,float]]
        Matrix
    b : tuple[tuple[float,float,float],tuple[float,float,float],tuple[float,float,float]]
        Matrix

    Returns
    -------
    c : tuple[tuple[float,float,float],tuple[float,float,float],tuple[float,float,float]]
        Matrix

    """

    return (
        (
            a[0][0] * b[0][0] + a[0][1] * b[1][0] + a[0][2] * b[2][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1] + a[0][2] * b[2][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2] * b[2][2],
        ),
        (
            a[1][0] * b[0][0] + a[1][1] * b[1][0] + a[1][2] * b[2][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1] + a[1][2] * b[2][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2] * b[2][2],
        ),
        (
            a[2][0] * b[0][0] + a[2][1] * b[1][0] + a[2][2] * b[2][0],
            a[2][0] * b[0][1] + a[2][1] * b[1][1] + a[2][2] * b[2][1],
            a[2][0] * b[0][2] + a[2][1] * b[1][2] + a[2][2] * b[2][2],
        ),
    )


@hjit("V(V,M)", inline=True)
def matmul_VM_hf(a, b):
    """
    Matmul (dot product) between a 3D row vector and a 3x3 matrix
    resulting in a 3D vector.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : tuple[tuple[float,float,float],tuple[float,float,float],tuple[float,float,float]]
        Matrix

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return (
        a[0] * b[0][0] + a[1] * b[1][0] + a[2] * b[2][0],
        a[0] * b[0][1] + a[1] * b[1][1] + a[2] * b[2][1],
        a[0] * b[0][2] + a[1] * b[1][2] + a[2] * b[2][2],
    )


@hjit("V(M,V)", inline=True)
def matmul_MV_hf(a, b):
    """
    Matmul (dot product) between a 3x3 matrix and a 3D column vector
    resulting in a 3D vector.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[tuple[float,float,float],tuple[float,float,float],tuple[float,float,float]]
        Matrix
    b : tuple[float,float,float]
        Vector

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return (
        b[0] * a[0][0] + b[1] * a[0][1] + b[2] * a[0][2],
        b[0] * a[1][0] + b[1] * a[1][1] + b[2] * a[1][2],
        b[0] * a[2][0] + b[1] * a[2][1] + b[2] * a[2][2],
    )


@hjit("f(V,V)", inline=True)
def matmul_VV_hf(a, b):
    """
    Matmul (dot product) between two 3D vectors resulting in a scalar.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : tuple[float,float,float]
        Vector

    Returns
    -------
    c : float
        Scalar

    """

    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@hjit("V(V,V)", inline=True)
def max_VV_hf(a, b):
    """
    Max elements element-wise from two 3D vectors.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : tuple[float,float,float]
        Vector

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return (
        a[0] if a[0] > b[0] else b[0],
        a[1] if a[1] > b[1] else b[1],
        a[2] if a[2] > b[2] else b[2],
    )


@hjit("V(V,f)", inline=True)
def mul_Vs_hf(a, b):
    """
    Multiplication of a 3D vector by a scalar element-wise.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : float
        Scalar

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return a[0] * b, a[1] * b, a[2] * b


@hjit("V(V,V)", inline=True)
def mul_VV_hf(a, b):
    """
    Multiplication of two 3D vectors element-wise.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : tuple[float,float,float]
        Vector

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return a[0] * b[0], a[1] * b[1], a[2] * b[2]


@hjit("f(V)", inline=True)
def norm_V_hf(a):
    """
    Norm of a 3D vector.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector

    Returns
    -------
    b : float
        Scalar

    """

    return sqrt(matmul_VV_hf(a, a))


@vjit("f(f,f,f)")
def norm_V_vf(a, b, c):
    """
    Norm of a 3D vector.

    Parameters
    ----------
    a : float
        First dimension scalar
    b : float
        Second dimension scalar
    c : float
        Third dimension scalar

    Returns
    -------
    d : float
        Scalar

    """

    return norm_V_hf((a, b, c))


@hjit("f(V,V)", inline=True)
def norm_VV_hf(a, b):
    """
    Combined norm of two 3D vectors treated like a single 6D vector.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : tuple[float,float,float]
        Vector

    Returns
    -------
    c : float
        Scalar

    """

    return sqrt(matmul_VV_hf(a, a) + matmul_VV_hf(b, b))


@hjit("f(f)", inline=True)
def sign_hf(a):
    """
    Sign of a float represented as another float (-1, 0, +1).
    Inline-compiled by default.

    Parameters
    ----------
    a : float
        Scalar

    Returns
    -------
    b : float
        Scalar

    """

    if a < 0.0:
        return -1.0
    if a == 0.0:
        return 0.0
    return 1.0  # if x > 0


@hjit("V(V,V)", inline=True)
def sub_VV_hf(a, b):
    """
    Subtraction of two 3D vectors element-wise.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[float,float,float]
        Vector
    b : tuple[float,float,float]
        Vector

    Returns
    -------
    c : tuple[float,float,float]
        Vector

    """

    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


@hjit("M(M)", inline=True)
def transpose_M_hf(a):
    """
    Transposition of a matrix.
    Inline-compiled by default.

    Parameters
    ----------
    a : tuple[tuple[float,float,float],tuple[float,float,float],tuple[float,float,float]]
        Matrix

    Returns
    -------
    b : tuple[tuple[float,float,float],tuple[float,float,float],tuple[float,float,float]]
        Matrix

    """

    return (
        (a[0][0], a[1][0], a[2][0]),
        (a[0][1], a[1][1], a[2][1]),
        (a[0][2], a[1][2], a[2][2]),
    )
