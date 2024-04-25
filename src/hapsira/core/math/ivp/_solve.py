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
    Compiles a dispatcher for a list of functions that can eventually called by index.

    Parameters
    ----------
    funcs : tuple[Callable, ...]
        One or multiple callables that require dispatching.
        Dispatching will be based on position in tuple.
        All callables must have the same signature.
    argtypes : argument portion of signature for callables
    restype : return type portion of signature for callables
    arguments : names of arguments for callables

    Returns
    -------
    b : Callable
        Dispatcher function

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
    """
    Find which event occurred during an integration step.

    Based on
    https://github.com/scipy/scipy/blob/4edfcaa3ce8a387450b6efce968572def71be089/scipy/integrate/_ivp/ivp.py#L130

    Parameters
    ----------
    g_old : float
        Value of event function at current point.
    g_new : float
        Value of event function at next point.
    direction : float
        Event "direction".

    Returns
    -------
    active : boolean
        Status of event (active or not)

    """

    up = (g_old <= 0) & (g_new >= 0)
    down = (g_old >= 0) & (g_new <= 0)
    either = up | down
    active = up & (direction > 0) | down & (direction < 0) | either & (direction == 0)
    return active
