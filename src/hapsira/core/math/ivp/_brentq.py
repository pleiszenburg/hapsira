import operator

import numpy as np


from ._solver import brentq_sf


_ECONVERGED = 0
_ESIGNERR = -1
_ECONVERR = -2
_EVALUEERR = -3
_EINPROGRESS = 1

CONVERGED = "converged"
SIGNERR = "sign error"
CONVERR = "convergence error"
VALUEERR = "value error"
INPROGRESS = "No error"

flag_map = {
    _ECONVERGED: CONVERGED,
    _ESIGNERR: SIGNERR,
    _ECONVERR: CONVERR,
    _EVALUEERR: VALUEERR,
    _EINPROGRESS: INPROGRESS,
}


def _wrap_nan_raise(f):
    def f_raise(x, *args):
        fx = f(x, *args)
        f_raise._function_calls += 1
        if np.isnan(fx):
            msg = f"The function value at x={x} is NaN; " "solver cannot continue."
            err = ValueError(msg)
            err._x = x
            err._function_calls = f_raise._function_calls
            raise err
        return fx

    f_raise._function_calls = 0
    return f_raise


_iter = 100
_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps


def brentq(
    f,
    a,
    b,
    args=(),
    xtol=_xtol,
    rtol=_rtol,
    maxiter=_iter,
):
    """
    Find a root of a function in a bracketing interval using Brent's method.

    Uses the classic Brent's method to find a root of the function `f` on
    the sign changing interval [a , b]. Generally considered the best of the
    rootfinding routines here. It is a safe version of the secant method that
    uses inverse quadratic extrapolation. Brent's method combines root
    bracketing, interval bisection, and inverse quadratic interpolation. It is
    sometimes known as the van Wijngaarden-Dekker-Brent method. Brent (1973)
    claims convergence is guaranteed for functions computable within [a,b].

    [Brent1973]_ provides the classic description of the algorithm. Another
    description can be found in a recent edition of Numerical Recipes, including
    [PressEtal1992]_. A third description is at
    http://mathworld.wolfram.com/BrentsMethod.html. It should be easy to
    understand the algorithm just by reading our code. Our code diverges a bit
    from standard presentations: we choose a different formula for the
    extrapolation step.

    Parameters
    ----------
    f : function
        Python function returning a number. The function :math:`f`
        must be continuous, and :math:`f(a)` and :math:`f(b)` must
        have opposite signs.
    a : scalar
        One end of the bracketing interval :math:`[a, b]`.
    b : scalar
        The other end of the bracketing interval :math:`[a, b]`.
    xtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be positive. For nice functions, Brent's
        method will often satisfy the above condition with ``xtol/2``
        and ``rtol/2``. [Brent1973]_
    rtol : number, optional
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter cannot be smaller than its default value of
        ``4*np.finfo(float).eps``. For nice functions, Brent's
        method will often satisfy the above condition with ``xtol/2``
        and ``rtol/2``. [Brent1973]_
    maxiter : int, optional
        If convergence is not achieved in `maxiter` iterations, an error is
        raised. Must be >= 0.
    args : tuple, optional
        Containing extra arguments for the function `f`.
        `f` is called by ``apply(f, (x)+args)``.
    full_output : bool, optional
        If `full_output` is False, the root is returned. If `full_output` is
        True, the return value is ``(x, r)``, where `x` is the root, and `r` is
        a `RootResults` object.
    disp : bool, optional
        If True, raise RuntimeError if the algorithm didn't converge.
        Otherwise, the convergence status is recorded in any `RootResults`
        return object.

    Returns
    -------
    root : float
        Root of `f` between `a` and `b`.
    r : `RootResults` (present if ``full_output = True``)
        Object containing information about the convergence. In particular,
        ``r.converged`` is True if the routine converged.

    Notes
    -----
    `f` must be continuous.  f(a) and f(b) must have opposite signs.

    Related functions fall into several classes:

    multivariate local optimizers
      `fmin`, `fmin_powell`, `fmin_cg`, `fmin_bfgs`, `fmin_ncg`
    nonlinear least squares minimizer
      `leastsq`
    constrained multivariate optimizers
      `fmin_l_bfgs_b`, `fmin_tnc`, `fmin_cobyla`
    global optimizers
      `basinhopping`, `brute`, `differential_evolution`
    local scalar minimizers
      `fminbound`, `brent`, `golden`, `bracket`
    N-D root-finding
      `fsolve`
    1-D root-finding
      `brenth`, `ridder`, `bisect`, `newton`
    scalar fixed-point finder
      `fixed_point`

    References
    ----------
    .. [Brent1973]
       Brent, R. P.,
       *Algorithms for Minimization Without Derivatives*.
       Englewood Cliffs, NJ: Prentice-Hall, 1973. Ch. 3-4.

    .. [PressEtal1992]
       Press, W. H.; Flannery, B. P.; Teukolsky, S. A.; and Vetterling, W. T.
       *Numerical Recipes in FORTRAN: The Art of Scientific Computing*, 2nd ed.
       Cambridge, England: Cambridge University Press, pp. 352-355, 1992.
       Section 9.3:  "Van Wijngaarden-Dekker-Brent Method."

    Examples
    --------
    >>> def f(x):
    ...     return (x**2 - 1)

    >>> from scipy import optimize

    >>> root = optimize.brentq(f, -2, 0)
    >>> root
    -1.0

    >>> root = optimize.brentq(f, 0, 2)
    >>> root
    1.0
    """
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError("xtol too small (%g <= 0)" % xtol)
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    f = _wrap_nan_raise(f)
    r = brentq_sf(f, a, b, xtol, rtol, maxiter, args)
    return r
