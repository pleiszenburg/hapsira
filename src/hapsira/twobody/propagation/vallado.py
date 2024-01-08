import sys

from astropy import units as u

from hapsira.core.propagation.vallado import vallado_rv_gf, VALLADO_NUMITER
from hapsira.twobody.propagation.enums import PropagatorKind
from hapsira.twobody.states import RVState

from ._compat import OldPropagatorModule

sys.modules[__name__].__class__ = OldPropagatorModule


class ValladoPropagator:
    """Propagates Keplerian orbit using Vallado's method.

    Notes
    -----
    This algorithm is based on Vallado implementation, and does basic Newton
    iteration on the Kepler equation written using universal variables. Battin
    claims his algorithm uses the same amount of memory but is between 40 %
    and 85 % faster.

    """

    kind = (
        PropagatorKind.ELLIPTIC | PropagatorKind.PARABOLIC | PropagatorKind.HYPERBOLIC
    )

    def __init__(self, numiter=VALLADO_NUMITER):
        self._numiter = numiter

    def propagate(self, state, tof):
        state = state.to_vectors()

        r_raw, v_raw = vallado_rv_gf(
            state.attractor.k.to_value(u.km**3 / u.s**2),
            *state.to_value(),
            tof.to_value(u.s),
            self._numiter,
        )
        r = r_raw << u.km
        v = v_raw << (u.km / u.s)

        new_state = RVState(state.attractor, (r, v), state.plane)
        return new_state
