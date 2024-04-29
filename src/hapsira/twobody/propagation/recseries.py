import sys

from astropy import units as u

from hapsira.core.propagation.recseries import (
    recseries_coe_vf,
    RECSERIES_METHOD_RTOL,
    RECSERIES_ORDER,
    RECSERIES_NUMITER,
    RECSERIES_RTOL,
)
from hapsira.twobody.propagation.enums import PropagatorKind
from hapsira.twobody.states import ClassicalState

from ._compat import OldPropagatorModule

sys.modules[__name__].__class__ = OldPropagatorModule


class RecseriesPropagator:
    """Kepler solver for elliptical orbits with recursive series approximation method.

    The order of the series is a user defined parameter.

    Notes
    -----
    This algorithm uses series discussed in the paper *Recursive solution to
    Kepler's problem for elliptical orbits - application in robust
    Newton-Raphson and co-planar closest approach estimation*
    with DOI: http://dx.doi.org/10.13140/RG.2.2.18578.58563/1

    """

    kind = PropagatorKind.ELLIPTIC

    def __init__(
        self,
        method=RECSERIES_METHOD_RTOL,
        order=RECSERIES_ORDER,
        numiter=RECSERIES_NUMITER,
        rtol=RECSERIES_RTOL,
    ):
        self._method = method
        self._order = order
        self._numiter = numiter
        self._rtol = rtol

    def propagate(self, state, tof):
        state = state.to_classical()

        nu = (
            recseries_coe_vf(
                state.attractor.k.to_value(u.km**3 / u.s**2),
                *state.to_value(),
                tof.to_value(u.s),
                self._method,
                self._order,
                self._numiter,
                self._rtol,
            )
            << u.rad
        )

        new_state = ClassicalState(
            state.attractor, state.to_tuple()[:5] + (nu,), state.plane
        )
        return new_state
