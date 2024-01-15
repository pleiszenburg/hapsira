from typing import Callable, Tuple
from warnings import warn

import numpy as np
from numba import jit

from . import _dop853_coefficients as dop853_coefficients
from ._dop853_coefficients import A, B, C

__all__ = [
    "EPS",
    "DOP853",
]


EPS = np.finfo(float).eps

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

INTERPOLATOR_POWER = 7
N_RV = 6
N_STAGES = 12
N_STAGES_EXTENDED = 16

A = A[:N_STAGES, :N_STAGES]
B = B
C = C[:N_STAGES]


@jit(nopython=False)
def norm(x: np.ndarray) -> float:
    """Compute RMS norm."""
    return np.linalg.norm(x) / x.size**0.5


@jit(nopython=False)
def rk_step(
    fun: Callable,
    t: float,
    y: np.ndarray,
    f: np.ndarray,
    h: float,
    K: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e., ``fun(x, y)``.
    h : float
        Step to use.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.
        The last row is a linear combination of the previous rows with
        coefficients

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.

    Const
    -----
    A : ndarray, shape (n_stages, n_stages)
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients at and above the main
        diagonal are zeros.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """

    assert y.shape == (N_RV,)
    assert f.shape == (N_RV,)
    assert K.shape == (N_STAGES + 1, N_RV)

    assert A.shape == (N_STAGES, N_STAGES)
    assert B.shape == (N_STAGES,)
    assert C.shape == (N_STAGES,)

    K[0] = f

    # for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
    #     dy = np.dot(K[:s].T, a[:s]) * h
    #     K[s] = fun(t + c * h, y + dy)

    dy = np.dot(K[:1].T, A[1, :1]) * h
    K[1] = fun(t + C[1] * h, y + dy)

    dy = np.dot(K[:2].T, A[2, :2]) * h
    K[2] = fun(t + C[2] * h, y + dy)

    dy = np.dot(K[:3].T, A[3, :3]) * h
    K[3] = fun(t + C[3] * h, y + dy)

    dy = np.dot(K[:4].T, A[4, :4]) * h
    K[4] = fun(t + C[4] * h, y + dy)

    dy = np.dot(K[:5].T, A[5, :5]) * h
    K[5] = fun(t + C[5] * h, y + dy)

    dy = np.dot(K[:6].T, A[6, :6]) * h
    K[6] = fun(t + C[6] * h, y + dy)

    dy = np.dot(K[:7].T, A[7, :7]) * h
    K[7] = fun(t + C[7] * h, y + dy)

    dy = np.dot(K[:8].T, A[8, :8]) * h
    K[8] = fun(t + C[8] * h, y + dy)

    dy = np.dot(K[:9].T, A[9, :9]) * h
    K[9] = fun(t + C[9] * h, y + dy)

    dy = np.dot(K[:10].T, A[10, :10]) * h
    K[10] = fun(t + C[10] * h, y + dy)

    dy = np.dot(K[:11].T, A[11, :11]) * h
    K[11] = fun(t + C[11] * h, y + dy)

    y_new = y + h * np.dot(K[:-1].T, B)
    f_new = fun(t + h, y_new)

    K[-1] = f_new

    assert y_new.shape == (N_RV,)
    assert f_new.shape == (N_RV,)

    return y_new, f_new


@jit(nopython=False)
def select_initial_step(
    fun: Callable,
    t0: float,
    y0: np.ndarray,
    f0: np.ndarray,
    direction: float,
    order: float,
    rtol: float,
    atol: float,
) -> float:
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    f0 : ndarray, shape (n,)
        Initial value of the derivative, i.e., ``fun(t0, y0)``.
    direction : float
        Integration direction.
    order : float
        Error estimator order. It means that the error controlled by the
        algorithm is proportional to ``step_size ** (order + 1)`.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    if y0.size == 0:
        return np.inf

    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    f1 = fun(t0 + h0 * direction, y1)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    return min(100 * h0, h1)


@jit(nopython=False)
def validate_max_step(max_step: float) -> float:
    """Assert that max_Step is valid and return it."""
    if max_step <= 0:
        raise ValueError("`max_step` must be positive.")
    return max_step


@jit(nopython=False)
def validate_tol(rtol: float, atol: float) -> Tuple[float, float]:
    """Validate tolerance values."""

    if np.any(rtol < 100 * EPS):
        warn(
            "At least one element of `rtol` is too small. "
            f"Setting `rtol = np.maximum(rtol, {100 * EPS})`."
        )
        rtol = np.maximum(rtol, 100 * EPS)

    atol = np.asarray(atol)
    if atol.ndim > 0 and atol.shape != (N_RV,):
        raise ValueError("`atol` has wrong shape.")

    if np.any(atol < 0):
        raise ValueError("`atol` must be positive.")

    return rtol, atol


class Dop853DenseOutput:
    """local interpolant over step made by an ODE solver.

    Attributes
    ----------
    t_min, t_max : float
        Time range of the interpolation.
    """

    def __init__(self, t_old, t, y_old, F):
        self.t_old = t_old
        self.t = t
        self.t_min = min(t, t_old)
        self.t_max = max(t, t_old)
        self.h = t - t_old
        self.F = F
        self.y_old = y_old

    def __call__(self, t):
        """Evaluate the interpolant.

        Parameters
        ----------
        t : float or array_like with shape (n_points,)
            Points to evaluate the solution at.

        Returns
        -------
        y : ndarray, shape (n,) or (n, n_points)
            Computed values. Shape depends on whether `t` was a scalar or a
            1-D array.
        """
        t = np.asarray(t)
        assert not t.ndim > 1
        return self._call_impl(t)

    def _call_impl(self, t):
        x = (t - self.t_old) / self.h

        if t.ndim == 0:
            y = np.zeros_like(self.y_old)
        else:
            x = x[:, None]
            y = np.zeros((len(x), len(self.y_old)), dtype=self.y_old.dtype)

        for i, f in enumerate(reversed(self.F)):
            y += f
            if i % 2 == 0:
                y *= x
            else:
                y *= 1 - x
        y += self.y_old

        return y.T


def check_arguments(fun, y0):
    """Helper function for checking arguments common to all solvers."""

    y0 = np.asarray(y0)

    assert not np.issubdtype(y0.dtype, np.complexfloating)

    dtype = float
    y0 = y0.astype(dtype, copy=False)

    assert not y0.ndim != 1
    assert np.isfinite(y0).all()

    def fun_wrapped(t, y):
        return np.asarray(fun(t, y), dtype=dtype)

    return fun_wrapped, y0


class DOP853:
    """Explicit Runge-Kutta method of order 8.

    This is a Python implementation of "DOP853" algorithm originally written
    in Fortran [1]_, [2]_. Note that this is not a literate translation, but
    the algorithmic core and coefficients are the same.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(t, y)``.
        Here, ``t`` is a scalar, and there are two options for the ndarray ``y``:
        It can either have shape (n,); then ``fun`` must return array_like with
        shape (n,). Alternatively it can have shape (n, k); then ``fun``
        must return an array_like with shape (n, k), i.e. each column
        corresponds to a single column in ``y``. The choice between the two
        options is determined by `vectorized` argument (see below).
    t0 : float
        Initial time.
    y0 : array_like, shape (n,)
        Initial state.
    t_bound : float
        Boundary time - the integration won't continue beyond it. It also
        determines the direction of the integration.
    max_step : float, optional
        Maximum allowed step size. Default is np.inf, i.e. the step size is not
        bounded and determined solely by the solver.
    rtol, atol : float and array_like, optional
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

    Attributes
    ----------
    n : int
        Number of equations.
    status : string
        Current status of the solver: 'running', 'finished' or 'failed'.
    t_bound : float
        Boundary time.
    direction : float
        Integration direction: +1 or -1.
    t : float
        Current time.
    y : ndarray
        Current state.
    t_old : float
        Previous time. None if no steps were made yet.
    step_size : float
        Size of the last successful step. None if no steps were made yet.
    nfev : int
        Number evaluations of the system's right-hand side.
    njev : int
        Number of evaluations of the Jacobian. Is always 0 for this solver
        as it does not use the Jacobian.
    nlu : int
        Number of LU decompositions. Is always 0 for this solver.
    """

    TOO_SMALL_STEP = "Required step size is less than spacing between numbers."

    order: int = 8
    error_estimator_order: int = 7
    E: np.ndarray = NotImplemented
    E3 = dop853_coefficients.E3
    E5 = dop853_coefficients.E5
    D = dop853_coefficients.D
    P: np.ndarray = NotImplemented

    A_EXTRA = dop853_coefficients.A[N_STAGES + 1 :]
    C_EXTRA = dop853_coefficients.C[N_STAGES + 1 :]

    def __init__(
        self,
        fun: Callable,
        t0: float,
        y0: np.array,
        t_bound: float,
        max_step: float = np.inf,
        rtol: float = 1e-3,
        atol: float = 1e-6,
    ):
        assert y0.shape == (N_RV,)

        self.t_old = None
        self.t = t0
        self._fun, self.y = check_arguments(fun, y0)
        self.t_bound = t_bound

        fun_single = self._fun

        def fun(t, y):
            self.nfev += 1
            return self.fun_single(t, y)

        self.fun = fun
        self.fun_single = fun_single

        self.direction = np.sign(t_bound - t0) if t_bound != t0 else 1
        self.status = "running"

        self.nfev = 0
        self.njev = 0
        self.nlu = 0

        self.y_old = None
        self.max_step = validate_max_step(max_step)
        self.rtol, self.atol = validate_tol(rtol, atol)
        self.f = self.fun(self.t, self.y)
        self.h_abs = select_initial_step(
            self.fun,
            self.t,
            self.y,
            self.f,
            self.direction,
            self.error_estimator_order,
            self.rtol,
            self.atol,
        )
        self.error_exponent = -1 / (self.error_estimator_order + 1)
        self.h_previous = None

        self.K_extended = np.empty((N_STAGES_EXTENDED, N_RV), dtype=self.y.dtype)
        self.K = self.K_extended[: N_STAGES + 1]

    @property
    def step_size(self):
        if self.t_old is None:
            return None
        else:
            return np.abs(self.t - self.t_old)

    def step(self):
        """Perform one integration step.

        Returns
        -------
        message : string or None
            Report from the solver. Typically a reason for a failure if
            `self.status` is 'failed' after the step was taken or None
            otherwise.
        """
        if self.status != "running":
            raise RuntimeError("Attempt to step on a failed or finished " "solver.")

        if self.t == self.t_bound:
            # Handle corner cases of empty solver or no integration.
            self.t_old = self.t
            self.t = self.t_bound
            message = None
            self.status = "finished"
        else:
            t = self.t
            success, message = self._step_impl()

            if not success:
                self.status = "failed"
            else:
                self.t_old = t
                if self.direction * (self.t - self.t_bound) >= 0:
                    self.status = "finished"

        return message

    def dense_output(self):
        """Compute a local interpolant over the last successful step.

        Returns
        -------
        sol : `DenseOutput`
            Local interpolant over the last successful step.
        """
        assert self.t_old is not None

        assert self.t != self.t_old

        return self._dense_output_impl()

    def _estimate_error_norm(self, K, h, scale):
        err5 = np.dot(K.T, self.E5) / scale
        err3 = np.dot(K.T, self.E3) / scale
        err5_norm_2 = np.linalg.norm(err5) ** 2
        err3_norm_2 = np.linalg.norm(err3) ** 2
        if err5_norm_2 == 0 and err3_norm_2 == 0:
            return 0.0
        denom = err5_norm_2 + 0.01 * err3_norm_2
        return np.abs(h) * err5_norm_2 / np.sqrt(denom * len(scale))

    def _step_impl(self):
        t = self.t
        y = self.y

        max_step = self.max_step
        rtol = self.rtol
        atol = self.atol

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        if self.h_abs > max_step:
            h_abs = max_step
        elif self.h_abs < min_step:
            h_abs = min_step
        else:
            h_abs = self.h_abs

        step_accepted = False
        step_rejected = False

        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP

            h = h_abs * self.direction
            t_new = t + h

            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound

            h = t_new - t
            h_abs = np.abs(h)

            y_new, f_new = rk_step(self.fun, t, y, self.f, h, self.K)
            scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
            error_norm = self._estimate_error_norm(self.K, h, scale)

            if error_norm < 1:
                if error_norm == 0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * error_norm**self.error_exponent)

                if step_rejected:
                    factor = min(1, factor)

                h_abs *= factor

                step_accepted = True
            else:
                h_abs *= max(MIN_FACTOR, SAFETY * error_norm**self.error_exponent)
                step_rejected = True

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new

        return True, None

    def _dense_output_impl(self):
        K = self.K_extended
        h = self.h_previous
        for s, (a, c) in enumerate(zip(self.A_EXTRA, self.C_EXTRA), start=N_STAGES + 1):
            dy = np.dot(K[:s].T, a[:s]) * h
            K[s] = self.fun(self.t_old + c * h, self.y_old + dy)

        F = np.empty((INTERPOLATOR_POWER, N_RV), dtype=self.y_old.dtype)

        f_old = K[0]
        delta_y = self.y - self.y_old

        F[0] = delta_y
        F[1] = h * f_old - delta_y
        F[2] = 2 * delta_y - h * (self.f + f_old)
        F[3:] = h * np.dot(self.D, K)

        return Dop853DenseOutput(self.t_old, self.t, self.y_old, F)
