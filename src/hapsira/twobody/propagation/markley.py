import sys

from astropy import units as u

from hapsira.core.propagation.markley import markley_coe_vf
from hapsira.twobody.propagation.enums import PropagatorKind
from hapsira.twobody.states import ClassicalState

from ._compat import OldPropagatorModule

sys.modules[__name__].__class__ = OldPropagatorModule


class MarkleyPropagator:
    """Elliptical Kepler Equation solver based on a fifth-order
    refinement of the solution of a cubic equation.

    Notes
    -----
    This method was originally presented by Markley in his paper *Kepler Equation Solver*
    with DOI: https://doi.org/10.1007/BF00691917

    """

    kind = PropagatorKind.ELLIPTIC

    def propagate(self, state, tof):
        state = state.to_classical()

        nu = (
            markley_coe_vf(
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
