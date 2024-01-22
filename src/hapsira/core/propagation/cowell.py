import numpy as np

from ..math.ivp import solve_ivp
from ..propagation.base import func_twobody_hf


__all__ = [
    "cowell",
]


def cowell(k, r, v, tofs, rtol=1e-11, events=None, f=func_twobody_hf):
    """
    Scalar cowell

    f : float
    r : ndarray (3,)
    v : ndarray (3,)
    tofs : ???
    rtol : float ... or also ndarray?
    """
    assert hasattr(f, "djit")  # DEBUG check for compiler flag
    assert isinstance(rtol, float)

    x, y, z = r
    vx, vy, vz = v

    u0 = np.array([x, y, z, vx, vy, vz])

    result = solve_ivp(
        f,
        (0, max(tofs)),
        u0,
        argk=k,
        rtol=rtol,
        atol=1e-12,
        # dense_output=True,
        events=events,
    )
    if not result.success:
        raise RuntimeError("Integration failed")

    if events is not None:
        # Collect only the terminal events
        terminal_events = [event for event in events if event.terminal]

        # If there are no terminal events, then the last time of integration is the
        # greatest one from the original array of propagation times
        if not terminal_events:
            last_t = max(tofs)
        else:
            # Filter the event which triggered first
            last_t = min(event._last_t for event in terminal_events)
            # FIXME: Here last_t has units, but tofs don't
            tofs = [tof for tof in tofs if tof < last_t] + [last_t]

    rrs = []
    vvs = []
    for i in range(len(tofs)):
        t = tofs[i]
        y = result.sol(t)
        rrs.append(y[:3])
        vvs.append(y[3:])

    return rrs, vvs
