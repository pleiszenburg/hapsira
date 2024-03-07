(coremodule)=
# Core module

The `core` module handles most actual heavy computations. It is compiled via [numba](https://numba.readthedocs.io). For both working with functions from `core` directly and contributing to it, it is highly recommended to gain some basic understanding of how `numba` works.

`core` is designed to work equally on CPUs and GPUs. (Most) exported `core` APIs interfacing with the rest of `hapsira` are either [universal functions](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-vectorize-decorator) or [generalized universal functions](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-guvectorize-decorator), which allow parallel operation with full [broadcasting semantics](https://numpy.org/doc/stable/user/basics.broadcasting.html).

```{warning}
Some `core` functions have yet not been refactored into this shape and will soon follow the same approach.
```

## Compiler targets

There are three compiler targets, which can be controlled through settings and/or environment variables:

- `cpu`: Single-threaded on CPUs
- `parallel`: Parallelized by `numba` via [threading layers](https://numba.readthedocs.io/en/stable/user/threading-layer.html) on CPUs
- `cuda`: Parallelized by `numba` via CUDA on Nvidia GPUs

All code of `core` will be compiled for one of the above listed targets. If multiple targets are supposed to be used simultaneously, this can only be achieved by multiple Python processes running in parallel.

## Compiler decorators

`core` offers the follwing JIT compiler decorators provided via `core.jit`:

- `vjit`: Wraps `numba.vectorize`. Functions decorated by it carry the suffix `_vf`.
- `gjit`: Wraps `numba.guvectorize`. Functions decorated by it carry the suffix `_gf`.
- `hjit`: Wraps `numba.jit` or `numba.cuda.jit`, depending on compiler target. Functions decorated by it carry the suffix `_hf`.
- `djit`: Variation of `hjit` with fixed function signature for user-provided functions used by `Cowell`

`core` functions dynamically generating (and compiling) functions within their scope carry `_hb`, `_vb` and `_gb` suffixes.

Wrapping `numba` functions allows to centralize compiler options and target switching as well as to simplify typing.

The decorators are applied in a **hierarchy**:

- Functions decorated by either `vjit` and `gjit` serve as the **only** interface between regular uncompiled Python code and `core`
- Functions decorated by `vjit` and `hjit` only call functions decorated by `hjit`
- Functions decorated by `hjit` can only call each other.

```{note}
The "hierarchy" of decorators is imposed by CUDA-compatibility. While functions decorated by `numba.jit` (targets `cpu` and `parallel`) can be called from uncompiled Python code, functions decorated by `numba.cuda.jit` (target `cuda`) are considered [device functions](https://numba.readthedocs.io/en/stable/cuda/device-functions.html) and can not be called by uncompiled Python code directly. They are supposed to be called by CUDA-kernels (or other device functions) only (slightly simplifying the actual situation as implemented by `numba`). If the target is set to `cuda`, functions decorated by `numba.vectorize` and `numba.guvectorize` become CUDA kernels.
```

```{warning}
As a result of name suffixes as of version `0.19.0`, many `core` module functions have been renamed making the package intentionally backwards-incompatible. Functions not yet using the new infrastructure can be recognized based on lack of suffix. Eventually all `core` functions will use this infrastructure and carry matching suffixes.
```

```{note}
Some functions decorated by `gjit` must receive a dummy parameter, also explicitly named `dummy`. It is usually an empty `numpy` array of shape `(3,)` of data type `u1` (unsigned one-byte integer). This is a work-around for [numba #2797](https://github.com/numba/numba/issues/2797).
```

## Compiler errors

Misconfigured compiler decorators or unavailable targets raise an `errors.JitError` exception.

## Keyword arguments and defaults

Due to incompletely documented limitations in `numba`, see [documentation](https://numba.readthedocs.io/en/stable/reference/pysupported.html#function-calls) and [numba #7870](https://github.com/numba/numba/issues/7870), functions decorated by `hjit`, `vjit` and `gjit` can not have defaults for any of their arguments. In this context, those functions can not reliably be called with keyword arguments, too, which must therefore be avoided. Defaults are provided as constants within the same submodule, usually the function name in capital letters followed by the name of the argument, also in capital letters.

## Dependencies

Functions decorated by `vjit`, `gjit` and `hjit` are only allowed to depend on Python's standard library's [math module](https://docs.python.org/3/library/math.html), but **not** on other third-party packages like [numpy](https://numpy.org/doc/stable/) or [scipy](https://docs.scipy.org/doc/scipy/) for that matter - except for certain details like [enforcing floating point precision](https://numpy.org/doc/stable/user/basics.types.html) as provided by `core.math.ieee754`

```{note}
Eliminating `numpy` and other dependencies serves two purposes. While it is critical for [CUDA-compatiblity](https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html), it additionally makes the code significantly faster on CPUs.
```

## Typing

All functions decorated by `hjit`, `vjit` and `gjit` must by typed using [signatures similar to those of numba](https://numba.readthedocs.io/en/stable/reference/types.html).

All compiled code enforces a single floating point precision level, which can be configured. The default is FP64 / double precision. For simplicity, the type shortcut is `f`, replacing `f2`, `f4` or `f8`. Consider the following example:

```python
from numba import vectorize
from hapsira.core.jit import vjit

@vectorize("f8(f8)")
def foo(x):
    return x ** 2

@vjit("f(f)")
def bar_vf(x):
    return x ** 2
```

Additional infrastructure can be found in `core.math.ieee754`. The default floating point type is exposed as `core.math.ieee754.float_` for explicit conversions. A matching epsilon is exposed as `core.math.ieee754.EPS`.

```{note}
Divisions by zero should, regardless of compiler target or even entirely deactivated compiler, always result in `inf` (infinity) instead of `ZeroDivisionError` exceptions. Most divisions within `core` are therefore explicitly guarded.
```

3D vectors are expressed as tuples, type shortcut `V`, replacing `Tuple([f,f,f])`. Consider the following example:

```python
from numba import njit
from hapsira.core.jit import hjit

@njit("f8(Tuple([f8,f8,f8]))")
def foo(x):
    return x[0] + x[1] + x[2]

@hjit("f(V)")
def bar_hf(x):
    return x[0] + x[1] + x[2]
```

Matrices are expressed as tuples of tuples, type shortcut `M`, replacing `Tuple([V,V,V])`. Consider the following example:

```python
from numba import njit
from hapsira.core.jit import hjit

@njit("f8(Tuple([Tuple([f8,f8,f8]),Tuple([f8,f8,f8]),Tuple([f8,f8,f8])]))")
def foo(x):
    sum_ = 0
    for idx in range(3):
        for jdx in range(3):
            sum_ += x[idx][jdx]
    return sum_

@hjit("f(M)")
def bar_hf(x):
    sum_ = 0
    for idx in range(3):
        for jdx in range(3):
            sum_ += x[idx][jdx]
    return sum_
```

Function types use the shortcut `F`, replacing `FunctionType`.

## Cowell’s formulation

Cowell’s formulation is one of the few places where `core` is exposed directly to the user.

### Two-body function

In its most simple form, the `CowellPropagator` relies on a variation of `func_twobody_hf` as a parameter, a function compiled by `hjit`, which can technically be omitted:

```python
from hapsira.core.propagation.base import func_twobody_hf
from hapsira.twobody.propagation import CowellPropagator

prop = CowellPropagator(f=func_twobody_hf)
prop = CowellPropagator()  # identical to the above
```

If perturbations are applied, however, `func_twobody_hf` needs to be altered. It is important that the new altered function is compiled via the `hjit` decorator and that is has the correct signature. To simplify the matter for users, a variation of `hjit` named `djit` carries the correct signature implicitly:

```python
from hapsira.core.jit import djit, hjit
from hapsira.core.math.linalg import mul_Vs_hf
from hapsira.core.propagation.base import func_twobody_hf
from hapsira.twobody.propagation import CowellPropagator

@hjit("Tuple([V,V])(f,V,V,f)")
def foo_hf(t0, rr, vv, k):
    du_kep_rr, du_kep_vv = func_twobody_hf(t0, rr, vv, k)
    return du_kep_rr, mul_Vs_hf(du_kep_vv, 1.1)  # multiply speed vector by 1.1

@djit
def bar_hf(t0, rr, vv, k):
    du_kep_rr, du_kep_vv = func_twobody_hf(t0, rr, vv, k)
    return du_kep_rr, mul_Vs_hf(du_kep_vv, 1.1)  # multiply speed vector by 1.1

prop = CowellPropagator(f=foo_hf)
prop = CowellPropagator(f=bar_hf)  # identical to the above
```

### Events

The core of each event's implementation must also be compiled by `hjit`. New events must inherit from `BaseEvent`. The compiled implementation should be an attribute, a function or a static method, named `_impl_hf`. Once this attribute is specified, an explicit call to the `_wrap` method, most likely from the constructor, automatically generates a second version of `_impl_hf` named `_impl_dense_hf` that is used to not only evaluate but also to approximate the exact time of flight of an event based on dense output of the underlying solver.

## Settings

The following settings, available via `settings.settings`, allow to alter the compiler's behaviour:

- `DEBUG`: `bool`, default `False`
- `CACHE`: `bool`, default `not DEBUG`
- `TARGET`: `str`, default `cpu`, alternatives `parallel` and `cuda`
- `INLINE`: `bool`, default `TARGET == "cuda"`
- `NOPYTHON`: `bool`, default `True`
- `FORCEOBJ`: `bool`, default `False`
- `PRECISION`: `str`, default `f8`, alternatives `f2` and `f4`

```{note}
Settings can be switched by either setting environment variables or importing the `settings` module **before** any other (sub-) module is imported.
```

The `DEBUG` setting disables caching and enables the highest log level, among other things.

`CACHE` only works for `cpu` and `parallel` targets. It speeds up import times drastically if the package gets reused. Dynamically generated functions can not be cached and must be exempt from caching by passing `cache = False` as a parameter to the JIT compiler decorator.

```{warning}
Building the cache should not be done in parallel processes - this will most likely result in non-deterministic segmentation faults, see [numba #4807](https://github.com/numba/numba/issues/4807). Once `core` is fully compiled and cached, it can however be used in parallel processes. Rebuilding the cache can usually reliably resolve segmentation faults.
```

Inlining via `INLINE` drastically increases performance but also compile times. It is the default behaviour for target `cuda`. See [relevant chapter in numba documentation](https://numba.readthedocs.io/en/stable/developer/inlining.html#notes-on-inlining) for details.

`NOPYTHON` and `FORCEOBJ` provide additional debugging capabilities but should not be changed for regular use. For details, see [nopython mode](https://numba.readthedocs.io/en/stable/glossary.html#term-nopython-mode) and [object mode](https://numba.readthedocs.io/en/stable/glossary.html#term-object-mode) in `numba`'s documentation.

The default `PRECISION` of all floating point operations is FP64 / double precision float.

```{warning}
`hapsira`, formerly `poliastro`, was validated for FP64. Certain parts like Cowell reliably operate at this precision only. Other parts like for instance atmospheric models can easily handle single precision. This option is therefore provided for experimental purposes only.
```

## Logging

Compiler issues are logged via logging channel `hapsira` using Python's standard library's [logging module](https://docs.python.org/3/howto/logging.html), also available as `debug.logger`. All compiler activity can be observed by enabling log level `debug`.

## Math

The former `_math` module, version `0.18` and earlier, has become a first-class citizen as `core.math`, fully compiled by the above mentioned infrastructure. `core.math` contains a number of replacements for `numpy` operations, mostly found in `core.math.linalg`. All of those functions do not allocate memory and are free of side-effects including a lack of changes to their parameters.

Functions in `core.math` follow a loose naming convention, indicating for what types of parameters they can be used. `mul_Vs_hf` for instance is a multiplication of a vector `V` and a scalar `s` (floating point). `M` indicates matricis.

`core.math` also replaces (some) required `scipy` functions:

- [scipy.interpolate.interp1d](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html) is replaced by `core.math.interpolate.interp_hb`. It custom-compiles 1D linear interpolators, embedding data statically into the compiled functions.
- [scipy.integrate.solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), [scipy.integrate.DOP853](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.DOP853.html) and [scipy.optimize.brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html) are replaced by `core.math.ivp`.

```{note}
Future releases might remove more dependencies to `scipy` from `core` for full CUDA compatibility and additional performance.
```
