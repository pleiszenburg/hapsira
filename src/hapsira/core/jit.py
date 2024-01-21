from typing import Callable, List, Union

import numba as nb
from numba import cuda

from hapsira.debug import logger
from hapsira.errors import JitError
from hapsira.settings import settings


__all__ = [
    "PRECISIONS",
    "hjit",
    "vjit",
    "gjit",
    "sjit",
    "array_to_V_hf",
]


logger.debug("jit target: %s", settings["TARGET"].value)
if settings["TARGET"].value == "cuda" and not cuda.is_available():
    raise JitError('selected target "cuda" is not available')

logger.debug("jit inline: %s", "yes" if settings["INLINE"].value else "no")
logger.debug("jit nopython: %s", "yes" if settings["NOPYTHON"].value else "no")

PRECISIONS = ("f4", "f8")  # TODO allow f2, i.e. half, for CUDA at least?


def _parse_signatures(signature: str, noreturn: bool = False) -> Union[str, List[str]]:
    """
    Automatically generate signatures for single and double
    """

    if "->" in signature:  # this is likely a layout for guvectorize
        logger.warning(
            "jit signature: likely a layout for guvectorize, not parsing (%s)",
            signature,
        )
        return signature

    if noreturn and not signature.startswith("void("):
        raise JitError(
            "function does not allow return values, likely compiled via guvectorize"
        )

    if not any(
        notation in signature for notation in ("f", "V", "M")
    ):  # leave this signature as it is
        logger.warning(
            "jit signature: no special notation, not parsing (%s)", signature
        )
        return signature

    if any(level in signature for level in PRECISIONS):  # leave this signature as it is
        logger.warning(
            "jit signature: precision specified, not parsing (%s)", signature
        )
        return signature

    signature = signature.replace("M", "Tuple([V,V,V])")  # matrix is a tuple of vectors
    signature = signature.replace("V", "Tuple([f,f,f])")  # vector is a tuple of floats
    # signature = signature.replace(
    #     "S", "Tuple([f,f,f,f,f,f])"
    # )  # state, two vectors, is a tuple of floats TODO remove, use vectors instead?
    assert "S" not in signature
    signature = signature.replace(
        "F", "FunctionType"
    )  # TODO does not work for CUDA yet

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

        if settings["TARGET"].value == "cuda":
            wjit = cuda.jit
            cfg = dict(
                device=True,
                inline=settings["INLINE"].value,
                cache=settings["CACHE"].value,
            )
        else:
            wjit = nb.jit
            cfg = dict(
                nopython=settings["NOPYTHON"].value,
                inline="always" if settings["INLINE"].value else "never",
                cache=settings["CACHE"].value,
            )
        cfg.update(kwargs)

        logger.debug(
            "hjit: func=%s, args=%s, kwargs=%s",
            getattr(inner_func, "__name__", repr(inner_func)),
            repr(args),
            repr(cfg),
        )

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
            target=settings["TARGET"].value,
            cache=settings["CACHE"].value,
        )
        if settings["TARGET"].value != "cuda":
            cfg["nopython"] = settings["NOPYTHON"].value
        cfg.update(kwargs)

        logger.debug(
            "vjit: func=%s, args=%s, kwargs=%s",
            getattr(inner_func, "__name__", repr(inner_func)),
            repr(args),
            repr(cfg),
        )

        return nb.vectorize(
            *args,
            **cfg,
        )(inner_func)

    if outer_func is not None:
        return wrapper(outer_func)

    return wrapper


def gjit(*args, **kwargs) -> Callable:
    """
    General vectorize on array, pre-configured, user-facing, switches compiler targets.
    Functions decorated by it can always be called directly if needed.
    """

    if len(args) == 1 and callable(args[0]):
        outer_func = args[0]
        args = tuple()
    else:
        outer_func = None

    if len(args) > 0 and isinstance(args[0], str):
        args = _parse_signatures(args[0], noreturn=True), *args[1:]

    def wrapper(inner_func: Callable) -> Callable:
        """
        Applies JIT
        """

        cfg = dict(
            target=settings["TARGET"].value,
            cache=settings["CACHE"].value,
        )
        if settings["TARGET"].value != "cuda":
            cfg["nopython"] = settings["NOPYTHON"].value
        cfg.update(kwargs)

        logger.debug(
            "gjit: func=%s, args=%s, kwargs=%s",
            getattr(inner_func, "__name__", repr(inner_func)),
            repr(args),
            repr(cfg),
        )

        return nb.guvectorize(
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
        outer_func = None

    if len(args) > 0 and isinstance(args[0], str):
        args = _parse_signatures(args[0]), *args[1:]

    def wrapper(inner_func: Callable) -> Callable:
        """
        Applies JIT
        """

        cfg = dict(
            nopython=settings["NOPYTHON"].value,
            inline="always" if settings["INLINE"].value else "never",
            **kwargs,
        )

        logger.debug(
            "sjit: func=%s, args=%s, kwargs=%s",
            getattr(inner_func, "__name__", repr(inner_func)),
            repr(args),
            repr(cfg),
        )

        return nb.jit(
            *args,
            **cfg,
        )(inner_func)

    if outer_func is not None:
        return wrapper(outer_func)

    return wrapper


@hjit("V(f[:])")
def array_to_V_hf(x):
    return x[0], x[1], x[2]
