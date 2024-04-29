import sys

from astropy import units as u
import numpy as np

from hapsira.core.math.ieee754 import float_
from hapsira.core.propagation.cowell import (
    cowell_gb,
    SOLVE_FINISHED,
    SOLVE_TERMINATED,
    SOLVE_BRENTQFAILED,
    SOLVE_FAILED,
)
from hapsira.core.propagation.base import func_twobody_hf
from hapsira.twobody.propagation.enums import PropagatorKind
from hapsira.twobody.states import RVState

from ._compat import OldPropagatorModule

sys.modules[__name__].__class__ = OldPropagatorModule


class CowellPropagator:
    """Propagates orbit using Cowell's formulation.

    Notes
    -----
    This method uses the Dormand & Prince integration method of order 8(5,3) (DOP853).
    If multiple tofs are provided, the method propagates to the maximum value
    (unless a terminal event is defined) and calculates the other values via dense output.

    """

    kind = (
        PropagatorKind.ELLIPTIC | PropagatorKind.PARABOLIC | PropagatorKind.HYPERBOLIC
    )

    def __init__(self, rtol=1e-11, atol=1e-12, events=tuple(), f=func_twobody_hf):
        self._rtol = rtol
        self._atol = atol
        self._events = events
        self._terminals = np.array([event.terminal for event in events], dtype=bool)
        self._directions = np.array([event.direction for event in events], dtype=float_)
        self._cowell_gf = cowell_gb(events=events, func=f)

    def propagate(self, state, tof):
        state = state.to_vectors()
        tofs = tof.reshape(-1)
        # TODO make sure tofs is sorted

        r0, v0 = state.to_value()
        (  # pylint: disable=E0633,E1120
            _,
            _,
            _,
            last_ts,
            status,
            t_idx,
            rrs,
            vvs,
        ) = self._cowell_gf(  # pylint: disable=E0633,E1120
            tofs.to_value(u.s),  # tofs
            r0,  # rr
            v0,  # vv
            state.attractor.k.to_value(u.km**3 / u.s**2),  # argk
            self._rtol,  # rtol
            self._atol,  # atol
            self._terminals,  # event_terminals
            self._directions,  # event_directions
        )

        assert np.all((status != SOLVE_FAILED))
        assert np.all((status != SOLVE_BRENTQFAILED))
        assert np.all((status == SOLVE_FINISHED) | (status == SOLVE_TERMINATED))

        for last_t, event in zip(last_ts, self._events):
            event.last_t_raw = last_t

        r = rrs[t_idx - 1] << u.km
        v = vvs[t_idx - 1] << (u.km / u.s)

        new_state = RVState(state.attractor, (r, v), state.plane)
        return new_state

    def propagate_many(self, state, tofs):
        state = state.to_vectors()
        # TODO make sure tofs is sorted

        r0, v0 = state.to_value()
        (  # pylint: disable=E0633,E1120
            _,
            _,
            _,
            last_ts,
            status,
            t_idx,
            rrs,
            vvs,
        ) = self._cowell_gf(  # pylint: disable=E0633,E1120
            tofs.to_value(u.s),  # tofs
            r0,  # rr
            v0,  # vv
            state.attractor.k.to_value(u.km**3 / u.s**2),  # argk
            self._rtol,  # rtol
            self._atol,  # atol
            self._terminals,  # event_terminals
            self._directions,  # event_directions
        )

        assert np.all((status != SOLVE_FAILED))
        assert np.all((status != SOLVE_BRENTQFAILED))
        assert np.all((status == SOLVE_FINISHED) | (status == SOLVE_TERMINATED))

        for last_t, event in zip(last_ts, self._events):
            event.last_t_raw = last_t

        # TODO: This should probably return a RVStateArray instead,
        # see discussion at https://github.com/poliastro/poliastro/pull/1492
        return (
            rrs[:t_idx, :] << u.km,
            vvs[:t_idx, :] << (u.km / u.s),
        )
