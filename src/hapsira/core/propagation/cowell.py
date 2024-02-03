import numpy as np

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

    Can be reversed: https://github.com/poliastro/poliastro/issues/1630
    """

    assert hasattr(f, "djit")  # DEBUG check for compiler flag
    assert isinstance(rtol, float)
    assert all(tof >= 0 for tof in tofs)
    assert sorted(tofs) == list(tofs)

    EVENTS = len(events)  # TODO compile as const

    event_impl_hfs = tuple(
        event.impl_hf for event in events
    )  # TODO compile into kernel
    event_impl_dense_hfs = tuple(
        event.impl_dense_hf for event in events
    )  # TODO compile into kernel
    event_terminals = np.array(
        [event.terminal for event in events], dtype=bool
    )  # gufunc param static
    event_directions = np.array(
        [event.direction for event in events], dtype=float
    )  # gufunc param static
    event_actives = np.full(
        (EVENTS,), fill_value=np.nan, dtype=bool
    )  # gufunc param TODO reset to nan
    event_g_olds = np.full(
        (EVENTS,), fill_value=np.nan, dtype=float
    )  # gufunc param TODO reset to nan
    event_g_news = np.full(
        (EVENTS,), fill_value=np.nan, dtype=float
    )  # gufunc param TODO reset to nan
    event_last_ts = np.full(
        (EVENTS,), fill_value=np.nan, dtype=float
    )  # gufunc param TODO reset to nan

    sol, success = solve_ivp(
        f,
        0.0,
        float(max(tofs)),
        array_to_V_hf(r),
        array_to_V_hf(v),
        argk=k,
        rtol=rtol,
        atol=atol,
        event_impl_hfs=event_impl_hfs,
        event_impl_dense_hfs=event_impl_dense_hfs,
        event_terminals=event_terminals,
        event_directions=event_directions,
        event_actives=event_actives,
        event_g_olds=event_g_olds,
        event_g_news=event_g_news,
        event_last_ts=event_last_ts,
    )
    if not success:
        raise RuntimeError("Integration failed")

    for idx in range(EVENTS):
        events[idx].last_t_raw = event_last_ts[idx]

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
