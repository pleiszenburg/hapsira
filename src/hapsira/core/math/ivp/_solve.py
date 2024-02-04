from math import nan, isnan
from typing import Callable, Tuple

import numpy as np

from ._const import DENSE_SIG
from ._brentq import brentq_dense_hf, BRENTQ_CONVERGED, BRENTQ_MAXITER
from ._rkcore import (
    dop853_init_hf,
    dop853_step_hf,
    DOP853_FINISHED,
    DOP853_FAILED,
    DOP853_ARGK,
    DOP853_FR,
    DOP853_FUN,
    DOP853_FV,
    DOP853_H_PREVIOUS,
    DOP853_K,
    DOP853_RR,
    DOP853_RR_OLD,
    DOP853_STATUS,
    DOP853_T,
    DOP853_T_OLD,
    DOP853_VV,
    DOP853_VV_OLD,
)
from ._rkdenseinterp import dop853_dense_interp_hf
from ._rkdenseoutput import dop853_dense_output_hf
from ..ieee754 import EPS
from ...jit import hjit


__all__ = [
    "solve_ivp",
]


SOLVE_RUNNING = -2
SOLVE_FAILED = -1
SOLVE_FINISHED = 0
SOLVE_TERMINATED = 1

TEMPLATE = """
@hjit("{RESTYPE:s}(i8,{ARGTYPES:s})", cache = False)
def dispatcher_hf(idx, {ARGUMENTS:s}):
{DISPATCHER:s}
    return {ERROR:s}
"""


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
def _event_is_active_hf(g_old, g_new, direction):
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


def solve_ivp(
    fun: Callable,
    tofs: np.ndarray,
    rr: Tuple[float, float, float],
    vv: Tuple[float, float, float],
    rrs: np.ndarray,
    vvs: np.ndarray,
    argk: float,
    rtol: float,
    atol: float,
    event_impl_hfs: Tuple[Callable, ...],
    event_impl_dense_hfs: Tuple[Callable, ...],
    event_terminals: np.ndarray,
    event_directions: np.ndarray,
    event_actives: np.ndarray,
    event_g_olds: np.ndarray,
    event_g_news: np.ndarray,
    event_last_ts: np.ndarray,
) -> Tuple[Callable, bool]:
    """
    Solve an initial value problem for a system of ODEs.
    """

    EVENTS = len(event_impl_hfs)  # TODO compile as const
    T0 = 0.0

    event_impl_hf = dispatcher_hb(
        funcs=event_impl_hfs,
        argtypes="f,V,V,f",
        restype="f",
        arguments="t, rr, vv, k",
    )
    event_impl_dense_hf = dispatcher_hb(
        funcs=event_impl_dense_hfs,
        argtypes=f"f,{DENSE_SIG:s},f",
        restype="f",
        arguments="t, t_old, h, rr_old, vv_old, F, argk",
    )

    solver = dop853_init_hf(fun, T0, rr, vv, tofs[-1], argk, rtol, atol)

    t_idx = 0
    t_last = T0

    for event_idx in range(EVENTS):
        event_g_olds[event_idx] = event_impl_hf(event_idx, T0, rr, vv, argk)
        event_last_ts[event_idx] = T0

    status = SOLVE_RUNNING
    while status == SOLVE_RUNNING:
        solver = dop853_step_hf(*solver)

        if solver[DOP853_STATUS] == DOP853_FINISHED:
            status = SOLVE_FINISHED
        elif solver[DOP853_STATUS] == DOP853_FAILED:
            status = SOLVE_FAILED
            break

        t_old = solver[DOP853_T_OLD]
        t = solver[DOP853_T]

        interpolant = dop853_dense_output_hf(
            solver[DOP853_FUN],
            solver[DOP853_ARGK],
            solver[DOP853_T_OLD],
            solver[DOP853_T],
            solver[DOP853_H_PREVIOUS],
            solver[DOP853_RR],
            solver[DOP853_VV],
            solver[DOP853_RR_OLD],
            solver[DOP853_VV_OLD],
            solver[DOP853_FR],
            solver[DOP853_FV],
            solver[DOP853_K],
        )

        at_least_one_active = False
        for event_idx in range(EVENTS):
            event_g_news[event_idx] = event_impl_hfs[event_idx](
                t, solver[DOP853_RR], solver[DOP853_VV], argk
            )
            event_last_ts[event_idx] = t
            event_actives[event_idx] = _event_is_active_hf(
                event_g_olds[event_idx],
                event_g_news[event_idx],
                event_directions[event_idx],
            )
            if event_actives[event_idx]:
                at_least_one_active = True

        if at_least_one_active:
            root_pivot = nan  # set initial value
            terminate = False

            for event_idx in range(EVENTS):
                if not event_actives[event_idx]:
                    continue

                if not event_terminals[event_idx]:
                    continue

                terminate = True

                event_last_ts[event_idx], root, brentq_status = brentq_dense_hf(
                    event_impl_dense_hf,
                    event_idx,
                    t_old,
                    t,
                    4 * EPS,
                    4 * EPS,
                    BRENTQ_MAXITER,
                    *interpolant,
                    argk,
                )
                if brentq_status != BRENTQ_CONVERGED:
                    return t_idx, False  # failed on event

                if isnan(root_pivot):
                    root_pivot = root
                    continue

                if t > t_old:  # smallest root of all active events
                    if root < root_pivot:
                        root_pivot = root
                    continue

                # largest root of all active events
                if root > root_pivot:
                    root_pivot = root
                raise ValueError("not t > t_old", t, t_old)  # TODO remove

            if terminate:
                assert not isnan(root_pivot)
                status = SOLVE_TERMINATED
                t = root_pivot

        for event_idx in range(EVENTS):
            event_g_olds[event_idx] = event_g_news[event_idx]

        if not t_last <= t:
            raise ValueError("not t_last <= t", t_last, t)

        while t_idx < tofs.shape[0] and tofs[t_idx] < t:
            rrs[t_idx, :], vvs[t_idx, :] = dop853_dense_interp_hf(
                tofs[t_idx], *interpolant
            )
            t_idx += 1
        if status == SOLVE_TERMINATED or tofs[t_idx] == t:
            rrs[t_idx, :], vvs[t_idx, :] = dop853_dense_interp_hf(t, *interpolant)
            t_idx += 1

        t_last = t

    return t_idx, status >= 0
