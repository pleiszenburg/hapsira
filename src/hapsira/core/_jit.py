from enum import Enum, auto
from typing import Callable
import os

import numba as nb
from numba import cuda

from hapsira.errors import JitError


__all__ = [
    "INLINE",
    "NOPYTHON",
    "PRECISIONS",
    "TARGET",
    "TARGETS",
    "hjit",
    "vjit",
    "sjit",
]


def _str2bool(value: str) -> bool:
    """
    Helper for parsing environment variables
    """

    if value.strip().lower() in ("true", "1", "yes"):
        return True
    if value.strip().lower() in ("false", "0", "no"):
        return False
    raise ValueError(f'can not convert value "{value:s}" to bool')


class TARGETS(Enum):
    """
    JIT targets
    """

    cpu = auto()
    parallel = auto()
    cuda = auto()
    # numba 0.54.0, 19 August 2021, removed AMD ROCm target

    @classmethod
    def get_default(cls):
        """
        Default JIT target
        """

        return cls.cpu

    @classmethod
    def get_current(cls):
        """
        Current JIT target
        """

        name = os.environ.get("HAPSIRA_TARGET", None)

        if name is None:
            target = cls.get_default()
        else:
            try:
                target = cls[name]
            except KeyError as e:
                raise JitError(
                    f'unknown target "{name:s}"; known targets are {repr(cls):s}'
                ) from e

        if target is cls.cuda and not cuda.is_available():
            raise JitError('selected target "cuda" is not available')

        return target


TARGET = TARGETS.get_current()

INLINE = _str2bool(
    os.environ.get("HAPSIRA_INLINE", "1" if TARGET is TARGET.cuda else "0")
)  # currently only relevant for helpers on cpu and parallel targets

NOPYTHON = _str2bool(
    os.environ.get("HAPSIRA_NOPYTHON", "1")
)  # only for debugging, True by default

PRECISIONS = ("f4", "f8")  # TODO allow f2, i.e. half, for CUDA at least?


def _parse_signatures(signature: str) -> str | list[str]:
    """
    Automatically generate signatures for single and double
    """

    if not any(
        notation in signature for notation in ("f", "V")
    ):  # leave this signature as it is
        # TODO warn!
        return signature

    if any(level in signature for level in PRECISIONS):  # leave this signature as it is
        # TODO warn!
        return signature

    signature = signature.replace(
        "V", "Tuple([f,f,f])"
    )  # TODO hope for support of "f[:]" return values in cuda target; 2D/4D vectors?
    return [signature.replace("f", dtype) for dtype in PRECISIONS]


def hjit(*args, **kwargs) -> Callable:
    """
    Scalar helper, pre-configured, internal, switches compiler targets.
    Functions decorated by it can only be called directly if TARGET is cpu or parallel.
    """

    if len(args) == 1 and callable(args[0]):
        outer_func = args[0]
        args = tuple()
    else:
        outer_func = None

    if len(args) > 0 and isinstance(args[0], str):
        args = _parse_signatures(args[0]), *args[1:]

    def wrapper(inner_func: Callable) -> Callable:
        """
        Applies JIT
        """

        if TARGET in (TARGETS.cpu, TARGETS.parallel):
            wjit = nb.jit
            cfg = dict(
                nopython=NOPYTHON,
                inline="always" if INLINE else "never",
            )
        elif TARGET is TARGETS.cuda:
            wjit = cuda.jit
            cfg = dict(
                device=True,
                inline=INLINE,
            )
        else:
            JitError(
                f'unknown target "{repr(TARGET):s}"; known targets are {repr(TARGETS):s}'
            )
        cfg.update(kwargs)

        return wjit(
            *args,
            **cfg,
        )(inner_func)

    if outer_func is not None:
        return wrapper(outer_func)

    return wrapper


def vjit(*args, **kwargs) -> Callable:
    """
    Vectorize on array, pre-configured, user-facing, switches compiler targets.
    Functions decorated by it can always be called directly if needed.
    """

    if len(args) == 1 and callable(args[0]):
        outer_func = args[0]
        args = tuple()
    else:
        outer_func = None

    if len(args) > 0 and isinstance(args[0], str):
        args = _parse_signatures(args[0]), *args[1:]

    def wrapper(inner_func: Callable) -> Callable:
        """
        Applies JIT
        """

        cfg = dict(
            target=TARGET.name,
        )
        if TARGET is not TARGETS.cuda:
            cfg["nopython"] = NOPYTHON
        cfg.update(kwargs)

        return nb.vectorize(
            *args,
            **cfg,
        )(inner_func)

    if outer_func is not None:
        return wrapper(outer_func)

    return wrapper


def sjit(*args, **kwargs) -> Callable:
    """
    Regular "scalar" (n)jit, pre-configured, potentially user-facing, always CPU compiler target.
    Functions decorated by it can always be called directly if needed.
    """

    if len(args) == 1 and callable(args[0]):
        outer_func = args[0]
        args = tuple()
    else:
        pass

    def wrapper(inner_func: Callable) -> Callable:
        """
        Applies JIT
        """

        cfg = dict(
            nopython=NOPYTHON,
            inline="always" if INLINE else "never",
        )
        cfg.update(kwargs)

        return nb.jit(
            *args,
            **cfg,
        )(inner_func)

    if outer_func is not None:
        return wrapper(outer_func)

    return wrapper
