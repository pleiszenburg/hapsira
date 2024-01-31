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
from ._rkdenseoutput import dense_output_hf
from ._solution import OdeSolution
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
    events: List[Callable],
    active_events,
    terminals,
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

    active_events = np.array(active_events)

    roots = [
        _solve_event_equation(events[event_index], interpolant, t_old, t, argk)
        for event_index in active_events
    ]

    roots = np.asarray(roots)

    if np.any(terminals[active_events]):
        if t > t_old:
            order = np.argsort(roots)
        else:
            order = np.argsort(-roots)
        active_events = active_events[order]
        roots = roots[order]
        t = np.nonzero(terminals[active_events])[0][0]
        roots = roots[: t + 1]
        terminate = True
    else:
        terminate = False

    return roots[-1], terminate


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
    events: Tuple[Callable],
) -> Tuple[OdeSolution, bool]:
    """
    Solve an initial value problem for a system of ODEs.
    """

    solver = dop853_init_hf(fun, t0, rr, vv, tf, argk, rtol, atol)
    ts = [t0]
    interpolants = []

    terminals = np.array([event.terminal for event in events])

    if len(events) > 0:
        gs_old = []
        for event in events:
            gs_old.append(event.impl_hf(t0, rr, vv, argk))
            event.last_t_raw = t0

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

        interpolant = dense_output_hf(
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

        if len(events) > 0:
            gs_new = []
            for event in events:
                gs_new.append(
                    event.impl_hf(t, solver[DOP853_RR], solver[DOP853_VV], argk)
                )
                event.last_t_raw = t

            actives = [
                _event_is_active_hf(g_old, g_new, event.direction)
                for g_old, g_new, event in zip(gs_old, gs_new, events)
            ]
            actives = [idx for idx, active in enumerate(actives) if active]

            if len(actives) > 0:
                root, terminate = _handle_events(
                    interpolant,
                    events,
                    actives,
                    terminals,
                    t_old,
                    t,
                    argk,
                )
                if terminate:
                    status = 1
                    t = root
            gs_old = gs_new

        assert ts[-1] <= t
        interpolants.append(interpolant)
        ts.append(t)

    return OdeSolution(np.array(ts), interpolants), status >= 0
