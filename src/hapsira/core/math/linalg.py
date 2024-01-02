from math import sqrt

from ..jit import hjit, vjit

__all__ = [
    "cross_VV_hf",
    "div_Vs_hf",
    "matmul_MM_hf",
    "matmul_VM_hf",
    "matmul_VV_hf",
    "mul_Vs_hf",
    "norm_hf",
    "norm_vf",
    "sub_VV_hf",
    "transpose_M_hf",
]


@hjit("V(V,V)")
def cross_VV_hf(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


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


@hjit("V(V,f)")
def mul_Vs_hf(v, s):
    return v[0] * s, v[1] * s, v[2] * s


@hjit("f(V)")
def norm_hf(a):
    return sqrt(matmul_VV_hf(a, a))


@vjit("f(f,f,f)")
def norm_vf(a, b, c):
    # TODO add axis setting in some way for util.norm?
    return norm_hf((a, b, c))


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
