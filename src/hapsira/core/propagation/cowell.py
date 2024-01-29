from ..jit import array_to_V_hf
from ..math.ivp import solve_ivp
from ..propagation.base import func_twobody_hf


__all__ = [
    "cowell",
]


def cowell(k, r, v, tofs, rtol=1e-11, atol=1e-12, events=tuple(), f=func_twobody_hf):
    """
    Scalar cowell

    k : float
    r : ndarray (3,)
    v : ndarray (3,)
    tofs : array of relative times [seconds]
    rtol : float
    atol : float
    events : Optional[List[Event]]
    f : Callable
    """

    assert hasattr(f, "djit")  # DEBUG check for compiler flag
    assert isinstance(rtol, float)

    sol, success = solve_ivp(
        f,
        0.0,
        float(max(tofs)),
        array_to_V_hf(r),
        array_to_V_hf(v),
        argk=k,
        rtol=rtol,
        atol=atol,
        events=tuple(events),
    )
    if not success:
        raise RuntimeError("Integration failed")

    if len(events) > 0:
        # Collect only the terminal events
        terminal_events = [event for event in events if event.terminal]

        # If there are no terminal events, then the last time of integration is the
        # greatest one from the original array of propagation times
        if len(terminal_events) > 0:
            # Filter the event which triggered first
            last_t = min(event.last_t_raw for event in terminal_events)
            tofs = [tof for tof in tofs if tof < last_t]
            tofs.append(last_t)

    rrs = []
    vvs = []
    for t in tofs:
        r_new, v_new = sol(t)
        rrs.append(r_new)
        vvs.append(v_new)

    return rrs, vvs
