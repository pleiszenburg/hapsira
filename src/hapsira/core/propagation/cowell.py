import numpy as np

from ..jit import array_to_V_hf
from ..math.ivp import solve_ivp
from ..propagation.base import func_twobody_hf


__all__ = [
    "cowell_vb",
]


def cowell_vb(
    k, r, v, tofs, rtol=1e-11, atol=1e-12, events=tuple(), func=func_twobody_hf
):
    """
    Scalar cowell

    k : float
    r : ndarray (3,)
    v : ndarray (3,)
    tofs : array of relative times [seconds]
    rtol : float
    atol : float
    events : Optional[List[Event]]
    func : Callable

    Can be reversed: https://github.com/poliastro/poliastro/issues/1630
    """

    assert hasattr(func, "djit")  # DEBUG check for compiler flag
    assert isinstance(rtol, float)
    assert all(tof >= 0 for tof in tofs)
    assert sorted(tofs) == list(tofs)

    EVENTS = len(events)  # TODO compile as const

    rrs = np.full((len(tofs), 3), fill_value=np.nan, dtype=float)
    vvs = np.full((len(tofs), 3), fill_value=np.nan, dtype=float)

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

    length, success = solve_ivp(
        func=func,
        tofs=tofs,
        rr=array_to_V_hf(r),
        vv=array_to_V_hf(v),
        rrs=rrs,
        vvs=vvs,
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

    return rrs[:length, :], vvs[:length, :]
