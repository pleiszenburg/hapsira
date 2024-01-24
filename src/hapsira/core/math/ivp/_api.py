import numpy as np

from ._brentq import brentq
from ._common import OdeSolution

from ._rk import EPS, DOP853


class OdeResult(dict):
    """Represents the optimization result.

    Attributes
    ----------
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    t, y, sol, t_events, y_events, nlu : ?
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys())


def solve_event_equation(event, sol, t_old, t):
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

    return brentq(lambda t: event(t, sol(t)), t_old, t, xtol=4 * EPS, rtol=4 * EPS)


def handle_events(sol, events, active_events, is_terminal, t_old, t):
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
        solve_event_equation(events[event_index], sol, t_old, t)
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


def prepare_events(events):
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


def find_active_events(g, g_new, direction):
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
    fun,
    t_span,
    y0,
    argk: float,
    method=DOP853,
    # t_eval=None,
    # dense_output=False,
    events=None,
    **options,
):
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
    method : string or `OdeSolver`, optional
        Integration method to use:
            * 'DOP853': Explicit Runge-Kutta method of order 8 [13]_.
              Python implementation of the "DOP853" algorithm originally
              written in Fortran [14]_. A 7-th order interpolation polynomial
              accurate to 7-th order is used for the dense output.
              Can be applied in the complex domain.
        You can also pass an arbitrary class derived from `OdeSolver` which
        implements the solver.
    events : callable, or list of callables, optional
        Events to track. If None (default), no events will be tracked.
        Each event occurs at the zeros of a continuous function of time and
        state. Each function must have the signature ``event(t, y)`` and return
        a float. The solver will find an accurate value of `t` at which
        ``event(t, y(t)) = 0`` using a root-finding algorithm. By default, all
        zeros will be found. The solver looks for a sign change over each step,
        so if multiple zero crossings occur within one step, events may be
        missed. Additionally each `event` function might have the following
        attributes:

            terminal: bool, optional
                Whether to terminate integration if this event occurs.
                Implicitly False if not assigned.
            direction: float, optional
                Direction of a zero crossing. If `direction` is positive,
                `event` will only trigger when going from negative to positive,
                and vice versa if `direction` is negative. If 0, then either
                direction will trigger event. Implicitly 0 if not assigned.

        You can assign attributes like ``event.terminal = True`` to any
        function in Python.
    **options
        Options passed to a chosen solver. All options available for already
        implemented solvers are listed below.
    first_step : float or None, optional
        Initial step size. Default is `None` which means that the algorithm
        should choose.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e., the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float or array_like, optional
        Relative and absolute tolerances. The solver keeps the local error
        estimates less than ``atol + rtol * abs(y)``. Here `rtol` controls a
        relative accuracy (number of correct digits), while `atol` controls
        absolute accuracy (number of correct decimal places). To achieve the
        desired `rtol`, set `atol` to be smaller than the smallest value that
        can be expected from ``rtol * abs(y)`` so that `rtol` dominates the
        allowable error. If `atol` is larger than ``rtol * abs(y)`` the
        number of correct digits is not guaranteed. Conversely, to achieve the
        desired `atol` set `rtol` such that ``rtol * abs(y)`` is always smaller
        than `atol`. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    jac : array_like, sparse_matrix, callable or None, optional
        Jacobian matrix of the right-hand side of the system with respect
        to y, required by the 'Radau', 'BDF' and 'LSODA' method. The
        Jacobian matrix has shape (n, n) and its element (i, j) is equal to
        ``d f_i / d y_j``.  There are three ways to define the Jacobian:

            * If array_like or sparse_matrix, the Jacobian is assumed to
              be constant. Not supported by 'LSODA'.
            * If callable, the Jacobian is assumed to depend on both
              t and y; it will be called as ``jac(t, y)``, as necessary.
              For 'Radau' and 'BDF' methods, the return value might be a
              sparse matrix.
            * If None (default), the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provide the Jacobian rather than
        relying on a finite-difference approximation.
    jac_sparsity : array_like, sparse matrix or None, optional
        Defines a sparsity structure of the Jacobian matrix for a finite-
        difference approximation. Its shape must be (n, n). This argument
        is ignored if `jac` is not `None`. If the Jacobian has only few
        non-zero elements in *each* row, providing the sparsity structure
        will greatly speed up the computations [10]_. A zero entry means that
        a corresponding element in the Jacobian is always zero. If None
        (default), the Jacobian is assumed to be dense.
        Not supported by 'LSODA', see `lband` and `uband` instead.
    lband, uband : int or None, optional
        Parameters defining the bandwidth of the Jacobian for the 'LSODA'
        method, i.e., ``jac[i, j] != 0 only for i - lband <= j <= i + uband``.
        Default is None. Setting these requires your jac routine to return the
        Jacobian in the packed format: the returned array must have ``n``
        columns and ``uband + lband + 1`` rows in which Jacobian diagonals are
        written. Specifically ``jac_packed[uband + i - j , j] = jac[i, j]``.
        The same format is used in `scipy.linalg.solve_banded` (check for an
        illustration).  These parameters can be also used with ``jac=None`` to
        reduce the number of Jacobian elements estimated by finite differences.
    min_step : float, optional
        The minimum allowed step size for 'LSODA' method.
        By default `min_step` is zero.

    Returns
    -------
    Bunch object with the following fields defined:
    t : ndarray, shape (n_points,)
        Time points.
    y : ndarray, shape (n, n_points)
        Values of the solution at `t`.
    sol : `OdeSolution` or None
        Found solution as `OdeSolution` instance; None if `dense_output` was
        set to False.
    t_events : list of ndarray or None
        Contains for each event type a list of arrays at which an event of
        that type event was detected. None if `events` was None.
    y_events : list of ndarray or None
        For each value of `t_events`, the corresponding value of the solution.
        None if `events` was None.
    nlu : int
        Number of LU decompositions.
    status : int
        Reason for algorithm termination:

            * -1: Integration step failed.
            *  0: The solver successfully reached the end of `tspan`.
            *  1: A termination event occurred.

    success : bool
        True if the solver reached the interval end or a termination event
        occurred (``status >= 0``).

    """

    t0, tf = map(float, t_span)

    solver = method(fun, t0, y0, tf, argk, **options)

    ts = [t0]
    ys = [y0]

    interpolants = []

    events, is_terminal, event_dir = prepare_events(events)

    if events is not None:
        events = [lambda t, x, event=event: event(t, x, argk) for event in events]
        g = [event(t0, y0) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
    else:
        t_events = None
        y_events = None

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
        y = solver.y

        sol = solver.dense_output()
        interpolants.append(sol)

        if events is not None:
            g_new = [event(t, y) for event in events]
            active_events = find_active_events(g, g_new, event_dir)
            if active_events.size > 0:
                if sol is None:
                    sol = solver.dense_output()

                root_indices, roots, terminate = handle_events(
                    sol, events, active_events, is_terminal, t_old, t
                )

                for e, te in zip(root_indices, roots):
                    t_events[e].append(te)
                    y_events[e].append(sol(te))

                if terminate:
                    status = 1
                    t = roots[-1]
                    y = sol(t)

            g = g_new

        ts.append(t)
        ys.append(y)

    if t_events is not None:
        t_events = [np.asarray(te) for te in t_events]
        y_events = [np.asarray(ye) for ye in y_events]

    ts = np.array(ts)
    ys = np.vstack(ys).T

    sol = OdeSolution(ts, interpolants)

    return OdeResult(
        t=ts,
        y=ys,
        sol=sol,
        t_events=t_events,
        y_events=y_events,
        status=status,
        success=status >= 0,
    )
