from scipy.integrate import DOP853, solve_ivp as solve_ivp_


def solve_ivp(*args, **kwargs):
    """
    Wrapper for `scipy.integrate.solve_ivp`
    """

    return solve_ivp_(
        *args,
        method=DOP853,
        **kwargs,
    )
