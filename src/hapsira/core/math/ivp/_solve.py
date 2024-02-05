from math import nan
from typing import Callable, Tuple

from ...jit import hjit


__all__ = [
    "event_is_active_hf",
    "dispatcher_hb",
]


TEMPLATE = """
@hjit("{RESTYPE:s}(i8,{ARGTYPES:s})", cache = False)
def dispatcher_hf(idx, {ARGUMENTS:s}):
{DISPATCHER:s}
    return {ERROR:s}
"""

_ = nan  # keep import alive


def dispatcher_hb(
    funcs: Tuple[Callable, ...],
    argtypes: str,
    restype: str,
    arguments: str,
    error: str = "nan",
) -> Callable:
    """
    Workaround for https://github.com/numba/numba/issues/9420
    """
    funcs = [
        (f"func_{id(func):x}", func) for func in funcs
    ]  # names are not unique, ids are
    globals_, locals_ = globals(), locals()  # HACK https://stackoverflow.com/a/71560563
    globals_.update({name: handle for name, handle in funcs})

    def switch(idx):
        return "if" if idx == 0 else "elif"

    code = TEMPLATE.format(
        DISPATCHER="\n".join(
            [
                f"    {switch(idx):s} idx == {idx:d}:\n        return {name:s}({arguments:s})"
                for idx, (name, _) in enumerate(funcs)
            ]
        ),  # TODO tree-like dispatch, faster
        ARGTYPES=argtypes,
        RESTYPE=restype,
        ARGUMENTS=arguments,
        ERROR=error,
    )
    exec(code, globals_, locals_)  # pylint: disable=W0122
    globals_["dispatcher_hf"] = locals_[
        "dispatcher_hf"
    ]  # HACK https://stackoverflow.com/a/71560563
    return dispatcher_hf  # pylint: disable=E0602  # noqa: F821


@hjit("b1(f,f,f)")
def event_is_active_hf(g_old, g_new, direction):
    """Find which event occurred during an integration step.

    Parameters
    ----------
    g, g_new : array_like, shape (n_events,)
        Values of event functions at a current and next points.
    directions : ndarray, shape (n_events,)
        Event "direction" according to the definition in `solve_ivp`.

    Returns
    -------
    active_events : ndarray
        Indices of events which occurred during the step.
    """
    up = (g_old <= 0) & (g_new >= 0)
    down = (g_old >= 0) & (g_new <= 0)
    either = up | down
    active = up & (direction > 0) | down & (direction < 0) | either & (direction == 0)
    return active
