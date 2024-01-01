from math import sqrt

from ..jit import hjit, vjit

__all__ = [
    "div_Vs_hf",
    "matmul_VV_hf",
    "mul_Vs_hf",
    "norm_hf",
    "norm_vf",
    "sub_VV_hf",
]


@hjit("V(V,f)")
def div_Vs_hf(v, s):
    return v[0] / s, v[1] / s, v[2] / s


@hjit("V(V,f)")
def mul_Vs_hf(v, s):
    return v[0] * s, v[1] * s, v[2] * s


@hjit("f(V,V)")
def matmul_VV_hf(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


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
