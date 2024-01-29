from typing import Callable, List, Optional, Tuple

import numpy as np

from ._brentq import brentq_dense_hf, BRENTQ_CONVERGED, BRENTQ_MAXITER
from ._solution import OdeSolution
from ._rk import DOP853
from ._rkdenseoutput import dense_output_hf
from ..ieee754 import EPS


__all__ = [
    "solve_ivp",
]


def _solve_event_equation(
    event: Callable, sol: Callable, t_old: float, t: float, argk: float
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
        *sol,
        argk,
    )
    event.last_t_raw = last_t
    assert BRENTQ_CONVERGED == status
    return value


def _handle_events(
    sol,
    events: List[Callable],
    active_events,
    is_terminal,
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
    is_terminal : ndarray, shape (n_events,)
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
    roots = [
        _solve_event_equation(events[event_index], sol, t_old, t, argk)
        for event_index in active_events
    ]

    roots = np.asarray(roots)

    if np.any(is_terminal[active_events]):
        if t > t_old:
            order = np.argsort(roots)
        else:
            order = np.argsort(-roots)
        active_events = active_events[order]
        roots = roots[order]
        t = np.nonzero(is_terminal[active_events])[0][0]
        active_events = active_events[: t + 1]
        roots = roots[: t + 1]
        terminate = True
    else:
        terminate = False

    return active_events, roots, terminate


def _prepare_events(events):
    """Standardize event functions and extract is_terminal and direction."""
    if callable(events):
        events = (events,)

    if events is not None:
        is_terminal = np.empty(len(events), dtype=bool)
        direction = np.empty(len(events))
        for i, event in enumerate(events):
            try:
                is_terminal[i] = event.terminal
            except AttributeError:
                is_terminal[i] = False

            try:
                direction[i] = event.direction
            except AttributeError:
                direction[i] = 0
    else:
        is_terminal = None
        direction = None

    return events, is_terminal, direction


def _find_active_events(g, g_new, direction):
    """Find which event occurred during an integration step.

    Parameters
    ----------
    g, g_new : array_like, shape (n_events,)
        Values of event functions at a current and next points.
    direction : ndarray, shape (n_events,)
        Event "direction" according to the definition in `solve_ivp`.

    Returns
    -------
    active_events : ndarray
        Indices of events which occurred during the step.
    """
    g, g_new = np.asarray(g), np.asarray(g_new)
    up = (g <= 0) & (g_new >= 0)
    down = (g >= 0) & (g_new <= 0)
    either = up | down
    mask = up & (direction > 0) | down & (direction < 0) | either & (direction == 0)

    return np.nonzero(mask)[0]


def solve_ivp(
    fun: Callable,
    t0: float,
    tf: float,
    rr: Tuple[float, float, float],
    vv: Tuple[float, float, float],
    argk: float,
    rtol: float,
    atol: float,
    events: Optional[List[Callable]] = None,
) -> Tuple[OdeSolution, bool]:
    """Solve an initial value problem for a system of ODEs.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system: the time derivative of the state ``y``
        at time ``t``. The calling signature is ``fun(t, y)``, where ``t`` is a
        scalar and ``y`` is an ndarray with ``len(y) = len(y0)``. ``fun`` must
        return an array of the same shape as ``y``. See `vectorized` for more
        information.
    t_span : 2-member sequence
        Interval of integration (t0, tf). The solver starts with t=t0 and
        integrates until it reaches t=tf. Both t0 and tf must be floats
        or values interpretable by the float conversion function.
    y0 : array_like, shape (n,)
        Initial state. For problems in the complex domain, pass `y0` with a
        complex data type (even if the initial value is purely real).
    events : callable, or list of callables, optional
        Events to track. If None (default), no events will be tracked.
        Each event occurs at the zeros of a continuous function of time and
        state. Each function must have the signature ``event(t, y)`` and return
        a float. The solver will find an accurate value of `t` at which
        ``event(t, y(t)) = 0`` using a root-finding algorithm. By default, all
        zeros will be found. The solver looks for a sign change over each step,
        so if multiple zero crossings occur within one step, events may be
        missed.
    """

    solver = DOP853(fun, t0, rr, vv, tf, argk, rtol, atol)

    ts = [t0]

    interpolants = []

    events, is_terminal, event_dir = _prepare_events(events)

    if events is not None:
        g = []
        for event in events:
            g.append(event.impl_hf(t0, rr, vv, argk))
            event.last_t_raw = t0

    status = None
    while status is None:
        solver.step()

        if solver.status == "finished":
            status = 0
        elif solver.status == "failed":
            status = -1
            break

        t_old = solver.t_old
        t = solver.t

        sol = dense_output_hf(
            solver.fun,
            solver.argk,
            solver.t_old,
            solver.t,
            solver.h_previous,
            solver.rr,
            solver.vv,
            solver.rr_old,
            solver.vv_old,
            solver.fr,
            solver.fv,
            solver.K,
        )
        interpolants.append(sol)

        if events is not None:
            g_new = []
            for event in events:
                g_new.append(event.impl_hf(t, solver.rr, solver.vv, argk))
                event.last_t_raw = t
            active_events = _find_active_events(g, g_new, event_dir)
            if active_events.size > 0:
                _, roots, terminate = _handle_events(
                    sol,
                    events,
                    active_events,
                    is_terminal,
                    t_old,
                    t,
                    argk,
                )
                if terminate:
                    status = 1
                    t = roots[-1]
            g = g_new

        ts.append(t)

    return OdeSolution(np.array(ts), interpolants), status >= 0
