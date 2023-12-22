from enum import Enum, auto
import os

import numba as nb
from poliastro.errors import JitError


class TARGETS(Enum):
    """
    JIT targets
    numba 0.54.0, 19 August 2021, removed AMD ROCm target
    """

    cpu = auto()
    parallel = auto()
    cuda = auto()

    @classmethod
    def get_default(cls):
        """
        default JIT target
        """

        return cls.cpu


TARGET = os.environ.get("HAPSIRA_TARGET", TARGETS.default().name)

try:
    TARGET = TARGETS[TARGET]
except KeyError as e:
    raise JitError(
        f'unknown target "{TARGET:s}"; known targets are {repr(TARGETS):s}'
    ) from e

if TARGET == "cuda":
    from numba import cuda

    if not cuda.is_available():
        raise JitError('no GPU for selected target "cuda" found')

INLINE = os.environ.get(
    "HAPSIRA_INLINE", "0"
)  # currently only relevant for helpers on cpu and parallel targets
if INLINE.lower() in ("true", "1", "yes"):
    INLINE = True
elif INLINE.lower() in ("false", "0", "no"):
    INLINE = False
else:
    raise ValueError(f'unknown value for inline "{INLINE:s}"')

PRECISIONS = ("f4", "f8")  # TODO allow f2, i.e. half, for CUDA at least?

# cuda.jit does not allow multiple signatures, i.e. stuff can only be compiled
# for one precision level, see https://github.com/numba/numba/issues/3226
CUDA_PRECISION = os.environ.get("HAPSIRA_CUDA_PRECISION", "f8")
if CUDA_PRECISION not in PRECISIONS:
    raise ValueError(f'unknown floating point precision "{CUDA_PRECISION:s}"')

NOPYTHON = True  # only for debugging, True by default


def _parse_signatures(signature):
    """
    Automatically generate signatures for single and double
    """
    if not any(
        notation in signature for notation in ("f", "V")
    ):  # leave this signature as it is
        return signature
    if any(level in signature for level in PRECISIONS):  # leave this signature as it is
        return signature
    signature = signature.replace(
        "V", "Tuple([f,f,f])"
    )  # TODO hope for support of "f[:]" return values in cuda target
    if TARGET == "cuda":
        return signature.replace("f", CUDA_PRECISION)
    return [signature.replace("f", dtype) for dtype in PRECISIONS]


def hjit(*args, **kwargs):
    """
    Scalar helper, pre-configured, internal, switches compiler targets.
    Functions decorated by it can only be called directly if TARGET is cpu or parallel.
    """

    if len(args) == 1 and callable(args[0]):
        func = args[0]
        args = tuple()
    else:
        func = None

    if len(args) > 0 and isinstance(args[0], str):
        args = _parse_signatures(args[0]), *args[1:]

    def wrapper(func):
        cfg = {}
        if TARGET in ("cpu", "parallel"):
            cfg.update(
                {"nopython": NOPYTHON, "inline": "always" if INLINE else "never"}
            )
        if TARGET == "cuda":
            cfg.update({"device": True, "inline": True})
        cfg.update(kwargs)

        wjit = cuda.jit if TARGET == "cuda" else nb.jit

        return wjit(
            *args,
            **cfg,
        )(func)

    if func is not None:
        return wrapper(func)

    return wrapper


def vjit(*args, **kwargs):
    """
    Vectorize on array, pre-configured, user-facing, switches compiler targets.
    Functions decorated by it can always be called directly if needed.
    """

    if len(args) == 1 and callable(args[0]):
        func = args[0]
        args = tuple()
    else:
        func = None

    if len(args) > 0 and isinstance(args[0], str):
        args = _parse_signatures(args[0]), *args[1:]

    def wrapper(func):
        cfg = {"target": TARGET}
        if TARGET in ("cpu", "parallel"):
            cfg.update({"nopython": NOPYTHON})
        cfg.update(kwargs)

        return nb.vectorize(
            *args,
            **cfg,
        )(func)

    if func is not None:
        return wrapper(func)

    return wrapper


def sjit(*args, **kwargs):
    """
    Regular "scalar" (n)jit, pre-configured, potentially user-facing, always CPU compiler target.
    Functions decorated by it can always be called directly if needed.
    """

    if len(args) == 1 and callable(args[0]):
        func = args[0]
        args = tuple()
    else:
        func = None

    def wrapper(func):
        cfg = {"nopython": NOPYTHON, "inline": "never"}  # inline in ('always', 'never')
        cfg.update(kwargs)

        return nb.jit(
            *args,
            **cfg,
        )(func)

    if func is not None:
        return wrapper(func)

    return wrapper


__all__ = [
    "CUDA_PRECISION",
    "INLINE",
    "NOPYTHON",
    "PRECISIONS",
    "TARGET",
    "hjit",
    "vjit",
    "sjit",
]
