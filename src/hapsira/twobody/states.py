from functools import cached_property

from astropy import units as u
import numpy as np

from hapsira.core.elements import (
    coe2mee_gf,
    coe2rv_gf,
    mee2coe_gf,
    mee2rv_gf,
    rv2coe_gf,
    RV2COE_TOL,
    mean_motion_vf,
    period_vf,
)
from hapsira.twobody.elements import t_p


u_km3s2 = u.km**3 / u.s**2


class BaseState:
    """Base State class, meant to be subclassed."""

    def __init__(self, attractor, elements, plane):
        """Constructor.

        Parameters
        ----------
        attractor : Body
            Main attractor.
        elements : tuple
            Six-tuple of orbital elements for this state.
        plane : ~hapsira.frames.enums.Planes
            Reference plane for the elements.

        """
        self._attractor = attractor
        self._elements = elements
        self._plane = plane

    @property
    def plane(self):
        """Fundamental plane of the frame."""
        return self._plane

    @property
    def attractor(self):
        """Main attractor."""
        return self._attractor

    @cached_property
    def n(self):
        """Mean motion."""
        return (
            mean_motion_vf(
                self.attractor.k.to_value(u_km3s2),
                self.to_classical().a.to_value(u.km),
            )
            * u.rad
            / u.s
        )

    @cached_property
    def period(self):
        """Period of the orbit."""
        return (
            period_vf(
                self.attractor.k.to_value(u_km3s2),
                self.to_classical().a.to_value(u.km),
            )
            * u.s
        )

    @cached_property
    def r_p(self):
        """Radius of pericenter."""
        return self.to_classical().a * (1 - self.to_classical().ecc)

    @cached_property
    def r_a(self):
        """Radius of apocenter."""
        return self.to_classical().a * (1 + self.to_classical().ecc)

    @cached_property
    def t_p(self):
        """Elapsed time since latest perifocal passage."""
        return t_p(
            self.to_classical().nu,
            self.to_classical().ecc,
            self.attractor.k,
            self.r_p,
        )

    def to_tuple(self):
        return self._elements

    def to_value(self):
        """Converts to raw values with appropriate units."""
        raise NotImplementedError

    def to_vectors(self):
        """Converts to position and velocity vector representation.

        Returns
        -------
        RVState

        """
        raise NotImplementedError

    def to_classical(self):
        """Converts to classical orbital elements representation.

        Returns
        -------
        ClassicalState

        """
        raise NotImplementedError

    def to_equinoctial(self):
        """Converts to modified equinoctial elements representation.

        Returns
        -------
        ModifiedEquinoctialState

        """
        raise NotImplementedError


class ClassicalState(BaseState):
    """State defined by its classical orbital elements.

    Orbital elements:

    p : ~astropy.units.Quantity
        Semilatus rectum.
    ecc : ~astropy.units.Quantity
        Eccentricity.
    inc : ~astropy.units.Quantity
        Inclination.
    raan : ~astropy.units.Quantity
        Right ascension of the ascending node.
    argp : ~astropy.units.Quantity
        Argument of the perigee.
    nu : ~astropy.units.Quantity
        True anomaly.

    """

    @property
    def p(self):
        """Semilatus rectum."""
        return self._elements[0]

    @property
    def a(self):
        """Semimajor axis."""
        return self.p / (1 - self.ecc**2)

    @property
    def ecc(self):
        """Eccentricity."""
        return self._elements[1]

    @property
    def inc(self):
        """Inclination."""
        return self._elements[2]

    @property
    def raan(self):
        """Right ascension of the ascending node."""
        return self._elements[3]

    @property
    def argp(self):
        """Argument of the perigee."""
        return self._elements[4]

    @property
    def nu(self):
        """True anomaly."""
        return self._elements[5]

    def to_value(self):
        return (
            self.p.to_value(u.km),
            self.ecc.value,
            self.inc.to_value(u.rad),
            self.raan.to_value(u.rad),
            self.argp.to_value(u.rad),
            self.nu.to_value(u.rad),
        )

    def to_vectors(self):
        """Converts to position and velocity vector representation."""

        r = np.zeros(self.attractor.k.shape + (3,), dtype=self.attractor.k.dtype)
        v = np.zeros(self.attractor.k.shape + (3,), dtype=self.attractor.k.dtype)

        coe2rv_gf(
            self.attractor.k.to_value(u.km**3 / u.s**2),
            *self.to_value(),
            np.zeros((3,), dtype="u1"),  # dummy
            r,
            v,
        )

        return RVState(self.attractor, (r << u.km, v << u.km / u.s), self.plane)

    def to_classical(self):
        """Converts to classical orbital elements representation."""
        return self

    def to_equinoctial(self):
        """Converts to modified equinoctial elements representation."""

        p, f, g, h, k, L = coe2mee_gf(*self.to_value())  # pylint: disable=E1120,E0633

        return ModifiedEquinoctialState(
            self.attractor,
            (
                p << u.km,
                f << u.rad,
                g << u.rad,
                h << u.rad,
                k << u.rad,
                L << u.rad,
            ),
            self.plane,
        )


class RVState(BaseState):
    """State defined by its position and velocity vectors.

    Orbital elements:

    r : ~astropy.units.Quantity
        Position vector wrt attractor center.
    v : ~astropy.units.Quantity
        Velocity vector.

    """

    @property
    def r(self):
        """Position vector."""
        return self._elements[0]

    @property
    def v(self):
        """Velocity vector."""
        return self._elements[1]

    def to_value(self):
        return (
            self.r.to_value(u.km),
            self.v.to_value(u.km / u.s),
        )

    def to_vectors(self):
        """Converts to position and velocity vector representation."""
        return self

    def to_classical(self):
        """Converts to classical orbital elements representation."""
        (p, ecc, inc, raan, argp, nu) = rv2coe_gf(  # pylint: disable=E1120,E0633
            self.attractor.k.to_value(u.km**3 / u.s**2),
            *self.to_value(),
            RV2COE_TOL,
        )

        return ClassicalState(
            self.attractor,
            (
                p << u.km,
                ecc << u.one,
                inc << u.rad,
                raan << u.rad,
                argp << u.rad,
                nu << u.rad,
            ),
            self.plane,
        )


class ModifiedEquinoctialState(BaseState):
    """State defined by modified equinoctial elements representation.

    Orbital elements:

    p : ~astropy.units.Quantity
        Semilatus rectum.
    f : ~astropy.units.Quantity
        Second modified equinoctial element.
    g : ~astropy.units.Quantity
        Third modified equinoctial element.
    h : ~astropy.units.Quantity
        Fourth modified equinoctial element.
    k : ~astropy.units.Quantity
        Fifth modified equinoctial element.
    L : ~astropy.units.Quantity
        True longitude.

    """

    @property
    def p(self):
        """Semilatus rectum."""
        return self._elements[0]

    @property
    def f(self):
        """Second modified equinoctial element."""
        return self._elements[1]

    @property
    def g(self):
        """Third modified equinoctial element."""
        return self._elements[2]

    @property
    def h(self):
        """Fourth modified equinoctial element."""
        return self._elements[3]

    @property
    def k(self):
        """Fifth modified equinoctial element."""
        return self._elements[4]

    @property
    def L(self):
        """True longitude."""
        return self._elements[5]

    def to_value(self):
        return (
            self.p.to_value(u.km),
            self.f.to_value(u.rad),
            self.g.to_value(u.rad),
            self.h.to_value(u.rad),
            self.k.to_value(u.rad),
            self.L.to_value(u.rad),
        )

    def to_classical(self):
        """Converts to classical orbital elements representation."""
        p, ecc, inc, raan, argp, nu = mee2coe_gf(  # pylint: disable=E1120,E0633
            *self.to_value()
        )

        return ClassicalState(
            self.attractor,
            (
                p << u.km,
                ecc << u.one,
                inc << u.rad,
                raan << u.rad,
                argp << u.rad,
                nu << u.rad,
            ),
            self.plane,
        )

    def to_vectors(self):
        """Converts to position and velocity vector representation."""
        r, v = mee2rv_gf(  # pylint: disable=E1120,E0633
            *self.to_value(), np.zeros((3,), dtype="u1")
        )
        return RVState(self.attractor, (r << u.km, v << u.km / u.s), self.plane)
