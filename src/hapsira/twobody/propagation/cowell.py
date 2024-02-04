import sys

from astropy import units as u

from hapsira.core.propagation.cowell import cowell_vb
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

    def __init__(self, rtol=1e-11, events=tuple(), f=func_twobody_hf):
        self._rtol = rtol
        self._events = events
        self._func = f

    def propagate(self, state, tof):
        state = state.to_vectors()
        tofs = tof.reshape(-1)

        rrs, vvs = cowell_vb(
            state.attractor.k.to_value(u.km**3 / u.s**2),
            *state.to_value(),
            tofs.to_value(u.s),
            self._rtol,
            events=self._events,
            func=self._func,
        )
        r = rrs[-1] << u.km
        v = vvs[-1] << (u.km / u.s)

        new_state = RVState(state.attractor, (r, v), state.plane)
        return new_state

    def propagate_many(self, state, tofs):
        state = state.to_vectors()

        rrs, vvs = cowell_vb(
            state.attractor.k.to_value(u.km**3 / u.s**2),
            *state.to_value(),
            tofs.to_value(u.s),
            self._rtol,
            events=self._events,
            func=self._func,
        )

        # TODO: This should probably return a RVStateArray instead,
        # see discussion at https://github.com/poliastro/poliastro/pull/1492
        return (
            rrs << u.km,
            vvs << (u.km / u.s),
        )
