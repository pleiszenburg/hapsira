from math import nan, isnan
from typing import Callable, List, Tuple

import numpy as np

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


def _solve_event_equation(
    event: Callable,
    interpolant: Callable,
    t_old: float,
    t: float,
    argk: float,
) -> float:
    """Solve an equation corresponding to an ODE event.

    The equation is ``event(t, y(t)) = 0``, here ``y(t)`` is known from an
    ODE solver using some sort of interpolation. It is solved by
    `scipy.optimize.brentq` with xtol=atol=4*EPS.

    Parameters
    ----------
    event : callable
        Function ``event(t, y)``.
    sol : callable
        Function ``sol(t)`` which evaluates an ODE solution between `t_old`
        and  `t`.
    t_old, t : float
        Previous and new values of time. They will be used as a bracketing
        interval.

    Returns
    -------
    root : float
        Found solution.
    """

    last_t, value, status = brentq_dense_hf(
        event.impl_dense_hf,
        t_old,
        t,
        4 * EPS,
        4 * EPS,
        BRENTQ_MAXITER,
        *interpolant,
        argk,
    )
    event.last_t_raw = last_t
    assert BRENTQ_CONVERGED == status
    return value


def _handle_events(
    interpolant,
    event_impl_dense_hfs: List[Callable],
    event_last_ts: np.ndarray,
    event_actives: np.ndarray,
    event_terminals: np.ndarray,
    t_old: float,
    t: float,
    argk: float,
):
    """Helper function to handle events.

    Parameters
    ----------
    sol : DenseOutput
        Function ``sol(t)`` which evaluates an ODE solution between `t_old`
        and  `t`.
    events : list of callables, length n_events
        Event functions with signatures ``event(t, y)``.
    active_events : ndarray
        Indices of events which occurred.
    terminals : ndarray, shape (n_events,)
        Which events are terminal.
    t_old, t : float
        Previous and new values of time.

    Returns
    -------
    root_indices : ndarray
        Indices of events which take zero between `t_old` and `t` and before
        a possible termination.
    roots : ndarray
        Values of t at which events occurred.
    terminate : bool
        Whether a terminal event occurred.
    """

    assert np.any(event_actives)  # nothing active

    EVENTS = len(event_impl_dense_hfs)  # TODO compile as const

    pivot = nan  # set initial value
    terminate = False

    for idx in range(EVENTS):
        if not event_actives[idx]:
            continue

        event_last_ts[idx], root, status = brentq_dense_hf(
            event_impl_dense_hfs[idx],
            t_old,
            t,
            4 * EPS,
            4 * EPS,
            BRENTQ_MAXITER,
            *interpolant,
            argk,
        )
        assert status == BRENTQ_CONVERGED

        if event_terminals[idx]:
            terminate = True

        if isnan(pivot):
            pivot = root
            continue

        if t > t_old:  # smallest root of all active events
            if root < pivot:
                pivot = root
            continue

        # largest root of all active events
        if root > pivot:
            pivot = root
        raise ValueError("not t > t_old", t, t_old)  # TODO remove

    assert not isnan(pivot)

    return pivot if terminate else nan, terminate


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
    t0: float,
    tf: float,
    rr: Tuple[float, float, float],
    vv: Tuple[float, float, float],
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

    solver = dop853_init_hf(fun, t0, rr, vv, tf, argk, rtol, atol)
    ts = [t0]
    interpolants = []

    _ = """
    Event:
        - impl_hf (callable) -> compiled tuple
        - impl_dense_hf (callable) -> compiled tuple
        - terminal (const) -> input array
        - direction (const) -> input array
        - is_active
        - g_old
        - g_new
        - last_t -> output array
    N events -> compiled const int?
    for-loop???
    """

    for idx in range(EVENTS):
        event_g_olds[idx] = event_impl_hfs[idx](t0, rr, vv, argk)
        event_last_ts[idx] = t0

    status = None
    while status is None:
        solver = dop853_step_hf(*solver)

        if solver[DOP853_STATUS] == DOP853_FINISHED:
            status = 0
        elif solver[DOP853_STATUS] == DOP853_FAILED:
            status = -1
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
        for idx in range(EVENTS):
            event_g_news[idx] = event_impl_hfs[idx](
                t, solver[DOP853_RR], solver[DOP853_VV], argk
            )
            event_last_ts[idx] = t
            event_actives[idx] = _event_is_active_hf(
                event_g_olds[idx],
                event_g_news[idx],
                event_directions[idx],
            )
            if event_actives[idx]:
                at_least_one_active = True

        if at_least_one_active:
            root, terminate = _handle_events(
                interpolant,
                event_impl_dense_hfs,  # TODO
                event_last_ts,
                event_actives,  # TODO
                event_terminals,  # TODO
                t_old,
                t,
                argk,
            )
            if terminate:
                status = 1
                t = root

        for idx in range(EVENTS):
            event_g_olds[idx] = event_g_news[idx]

        if not ts[-1] <= t:
            raise ValueError("not ts[-1] <= t", ts[-1], t)
        interpolants.append(interpolant)
        ts.append(t)

    assert len(ts) >= 2
    assert len(ts) == len(interpolants) + 1
    assert (
        (len(ts) == 2 and ts[0] == ts[1])
        or all(a - b > 0 for a, b in zip(ts[:-1], ts[1:]))
        or all(b - a > 0 for a, b in zip(ts[:-1], ts[1:]))
    )

    def ode_solution(
        t: float,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Evaluate the solution
        """
        idx = np.searchsorted(ts, t, side="left")
        segment = min(max(idx - 1, 0), len(interpolants) - 1)
        return dop853_dense_interp_hf(t, *interpolants[segment])

    return ode_solution, status >= 0
