import sys

from astropy import units as u

from hapsira.core.propagation.farnocchia import (
    farnocchia_coe_vf,
    farnocchia_rv_gf,
)
from hapsira.twobody.propagation.enums import PropagatorKind
from hapsira.twobody.states import ClassicalState

from ._compat import OldPropagatorModule

sys.modules[__name__].__class__ = OldPropagatorModule


class FarnocchiaPropagator:
    r"""Propagates orbit using Farnocchia's method.

    Notes
    -----
    This method takes initial :math:`\vec{r}, \vec{v}`, calculates classical orbit parameters,
    increases mean anomaly and performs inverse transformation to get final :math:`\vec{r}, \vec{v}`
    The logic is based on formulae (4), (6) and (7) from http://dx.doi.org/10.1007/s10569-013-9476-9

    """

    kind = (
        PropagatorKind.ELLIPTIC | PropagatorKind.PARABOLIC | PropagatorKind.HYPERBOLIC
    )

    def propagate(self, state, tof):
        state = state.to_classical()

        nu = (
            farnocchia_coe_vf(
                state.attractor.k.to_value(u.km**3 / u.s**2),
                *state.to_value(),
                tof.to_value(u.s),
            )
            << u.rad
        )

        new_state = ClassicalState(
            state.attractor, state.to_tuple()[:5] + (nu,), state.plane
        )
        return new_state

    def propagate_many(self, state, tofs):
        state = state.to_vectors()
        k = state.attractor.k.to_value(u.km**3 / u.s**2)
        rv0 = state.to_value()

        # TODO: This should probably return a ClassicalStateArray instead,
        # see discussion at https://github.com/hapsira/hapsira/pull/1492
        rr, vv = farnocchia_rv_gf(k, *rv0, tofs.to_value(u.s))  # pylint: disable=E0633

        return (
            rr << u.km,
            vv << (u.km / u.s),
        )
