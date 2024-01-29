from math import inf, sqrt

from .ieee754 import EPS
from ..jit import hjit, vjit

__all__ = [
    "abs_V_hf",
    "add_Vs_hf",
    "add_VV_hf",
    "cross_VV_hf",
    "div_Vs_hf",
    "div_VV_hf",
    "matmul_MM_hf",
    "matmul_VM_hf",
    "matmul_VV_hf",
    "max_VV_hf",
    "mul_Vs_hf",
    "mul_VV_hf",
    "nextafter_hf",
    "norm_V_hf",
    "norm_V_vf",
    "norm_VV_hf",
    "sign_hf",
    "sub_VV_hf",
    "transpose_M_hf",
]


@hjit("V(V)")
def abs_V_hf(x):
    return abs(x[0]), abs(x[1]), abs(x[2])


@hjit("V(V,f)")
def add_Vs_hf(a, b):
    return a[0] + b, a[1] + b, a[2] + b


@hjit("V(V,V)")
def add_VV_hf(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


@hjit("V(V,V)")
def cross_VV_hf(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@hjit("f(f,f)")
def div_ss_hf(a, b):
    """
    Similar to np.divide
    """
    if b == 0:
        return inf if a >= 0 else -inf
    return a / b


@hjit("V(V,V)")
def div_VV_hf(x, y):
    return x[0] / y[0], x[1] / y[1], x[2] / y[2]


@hjit("V(V,f)")
def div_Vs_hf(v, s):
    return v[0] / s, v[1] / s, v[2] / s


@hjit("M(M,M)")
def matmul_MM_hf(a, b):
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


@hjit("V(V,M)")
def matmul_VM_hf(a, b):
    return (
        a[0] * b[0][0] + a[1] * b[1][0] + a[2] * b[2][0],
        a[0] * b[0][1] + a[1] * b[1][1] + a[2] * b[2][1],
        a[0] * b[0][2] + a[1] * b[1][2] + a[2] * b[2][2],
    )


@hjit("f(V,V)")
def matmul_VV_hf(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@hjit("V(V,V)")
def max_VV_hf(x, y):
    return (
        x[0] if x[0] > y[0] else y[0],
        x[1] if x[1] > y[1] else y[1],
        x[2] if x[2] > y[2] else y[2],
    )


@hjit("V(V,f)")
def mul_Vs_hf(v, s):
    return v[0] * s, v[1] * s, v[2] * s


@hjit("V(V,V)")
def mul_VV_hf(a, b):
    return a[0] * b[0], a[1] * b[1], a[2] * b[2]


@hjit("f(f,f)")
def nextafter_hf(x, direction):
    if x < direction:
        return x + EPS
    return x - EPS


@hjit("f(V)")
def norm_V_hf(a):
    return sqrt(matmul_VV_hf(a, a))


@vjit("f(f,f,f)")
def norm_V_vf(a, b, c):
    # TODO add axis setting in some way for util.norm?
    return norm_V_hf((a, b, c))


@hjit("f(V,V)")
def norm_VV_hf(x, y):
    return sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + y[0] ** 2 + y[1] ** 2 + y[2] ** 2)


@hjit("f(f)")
def sign_hf(x):
    if x < 0.0:
        return -1.0
    if x == 0.0:
        return 0.0
    return 1.0  # if x > 0


@hjit("V(V,V)")
def sub_VV_hf(va, vb):
    return va[0] - vb[0], va[1] - vb[1], va[2] - vb[2]


@hjit("M(M)")
def transpose_M_hf(a):
    return (
        (a[0][0], a[1][0], a[2][0]),
        (a[0][1], a[1][1], a[2][1]),
        (a[0][2], a[1][2], a[2][2]),
    )
