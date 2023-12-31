from math import sqrt

from ..jit import hjit, vjit

__all__ = [
    "matmul_VV_hf",
    "norm_hf",
    "norm_vf",
]


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
