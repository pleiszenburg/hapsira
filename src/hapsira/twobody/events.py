from abc import ABC, abstractmethod
from warnings import warn

from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel
import numpy as np

from hapsira.core.jit import array_to_V_hf
from hapsira.core.math.linalg import norm_V_vf
from hapsira.core.events import (
    eclipse_function_hf,
    line_of_sight_gf,
)
from hapsira.core.math.interpolate import interp_hb
from hapsira.core.spheroid_location import cartesian_to_ellipsoidal_hf
from hapsira.util import time_range


__all__ = [
    "BaseEvent",
    "AltitudeCrossEvent",
    "LithobrakeEvent",
    "LatitudeCrossEvent",
    "BaseEclipseEvent",
    "PenumbraEvent",
    "UmbraEvent",
    "NodeCrossEvent",
    "LosEvent",
]


class BaseEvent(ABC):
    """Base class for event functionalities.

    Parameters
    ----------
    terminal: bool
        Whether to terminate integration if this event occurs.
    direction: float
        Handle triggering of event.

    """

    def __init__(self, terminal, direction):
        self._terminal, self._direction = terminal, direction
        self._last_t = None

    @property
    def terminal(self):
        return self._terminal

    @property
    def direction(self):
        return self._direction

    @property
    def last_t(self):
        return self._last_t << u.s

    @abstractmethod
    def __call__(self, t, u, k):
        raise NotImplementedError


class AltitudeCrossEvent(BaseEvent):
    """Detect if a satellite crosses a specific threshold altitude.

    Parameters
    ----------
    alt: float
        Threshold altitude (km).
    R: float
        Radius of the attractor (km).
    terminal: bool
        Whether to terminate integration if this event occurs.
    direction: float
        Handle triggering of event based on whether altitude is crossed from above
        or below, defaults to -1, i.e., event is triggered only if altitude is
        crossed from above (decreasing altitude).

    """

    def __init__(self, alt, R, terminal=True, direction=-1):
        super().__init__(terminal, direction)
        self._R = R
        self._alt = alt  # Threshold altitude from the ground.

    def __call__(self, t, u, k):
        self._last_t = t
        r_norm = norm_V_vf(*u[:3])

        return (
            r_norm - self._R - self._alt
        )  # If this goes from +ve to -ve, altitude is decreasing.


class LithobrakeEvent(AltitudeCrossEvent):
    """Terminal event that detects impact with the attractor surface.

    Parameters
    ----------
    R : float
        Radius of the attractor (km).
    terminal: bool
        Whether to terminate integration if this event occurs.

    """

    def __init__(self, R, terminal=True):
        super().__init__(0, R, terminal, direction=-1)


class LatitudeCrossEvent(BaseEvent):
    """Detect if a satellite crosses a specific threshold latitude.

    Parameters
    ----------
    orbit: ~hapsira.twobody.orbit.Orbit
        Orbit.
    lat: astropy.quantity.Quantity
        Threshold latitude.
    terminal: bool, optional
        Whether to terminate integration if this event occurs, defaults to True.
    direction: float, optional
        Handle triggering of event based on whether latitude is crossed from above
        or below, defaults to 0, i.e., event is triggered while traversing from both directions.

    """

    def __init__(self, orbit, lat, terminal=False, direction=0):
        super().__init__(terminal, direction)

        self._R = orbit.attractor.R.to_value(u.m)
        self._R_polar = orbit.attractor.R_polar.to_value(u.m)
        self._epoch = orbit.epoch
        self._lat = lat.to_value(u.deg)  # Threshold latitude (in degrees).

    def __call__(self, t, u_, k):
        self._last_t = t
        pos_on_body = (u_[:3] / norm_V_vf(*u_[:3])) * self._R
        _, lat_, _ = cartesian_to_ellipsoidal_hf(
            self._R, self._R_polar, *pos_on_body
        )  # TODO call into hf

        return np.rad2deg(lat_) - self._lat


class BaseEclipseEvent(BaseEvent):
    """Base class for the eclipse event.

    Parameters
    ----------
    orbit: hapsira.twobody.orbit.Orbit
        Orbit of the satellite.
    tof: ~astropy.units.Quantity
        Maximum time of flight for interpolator
    steps: int
        Steps for interpolator
    terminal: bool, optional
        Whether to terminate integration when the event occurs, defaults to False.
    direction: float, optional
        Specify which direction must the event trigger, defaults to 0.

    """

    def __init__(self, orbit, tof, steps=50, terminal=False, direction=0):
        super().__init__(terminal, direction)
        self._primary_body = orbit.attractor
        self._secondary_body = orbit.attractor.parent
        self._epoch = orbit.epoch
        self.k = self._primary_body.k.to_value(u.km**3 / u.s**2)
        self.R_sec = self._secondary_body.R.to_value(u.km)
        self.R_primary = self._primary_body.R.to_value(u.km)

        epochs = time_range(start=self._epoch, end=self._epoch + tof, num_values=steps)
        r_primary_wrt_ssb, _ = get_body_barycentric_posvel(
            self._primary_body.name, epochs
        )
        r_secondary_wrt_ssb, _ = get_body_barycentric_posvel(
            self._secondary_body.name, epochs
        )
        self._r_sec_hf = interp_hb(
            (epochs - self._epoch).to_value(u.s),
            (r_secondary_wrt_ssb - r_primary_wrt_ssb).xyz.to_value(u.km),
        )

    @abstractmethod
    def __call__(self, t, u_, k):
        # Solve for primary and secondary bodies position w.r.t. solar system
        # barycenter at a particular epoch.
        return self._r_sec_hf(t)


class PenumbraEvent(BaseEclipseEvent):
    """Detect whether a satellite is in penumbra or not.

    Parameters
    ----------
    orbit: hapsira.twobody.orbit.Orbit
        Orbit of the satellite.
    terminal: bool, optional
        Whether to terminate integration when the event occurs, defaults to False.
    direction: float, optional
        Handle triggering of event based on whether entry is into or out of
        penumbra, defaults to 0, i.e., event is triggered at both, entry and exit points.

    """

    def __call__(self, t, u_, k):
        self._last_t = t

        r_sec = super().__call__(t, u_, k)
        shadow_function = eclipse_function_hf(
            self.k,
            array_to_V_hf(u_[:3]),
            array_to_V_hf(u_[3:]),
            r_sec,
            self.R_sec,
            self.R_primary,
            False,
        )  # TODO call into hf

        return shadow_function


class UmbraEvent(BaseEclipseEvent):
    """Detect whether a satellite is in umbra or not.

    Parameters
    ----------
    orbit: hapsira.twobody.orbit.Orbit
        Orbit of the satellite.
    terminal: bool, optional
        Whether to terminate integration when the event occurs, defaults to False.
    direction: float, optional
        Handle triggering of event based on whether entry is into or out of
        umbra, defaults to 0, i.e., event is triggered at both, entry and exit points.

    """

    def __call__(self, t, u_, k):
        self._last_t = t

        r_sec = super().__call__(t, u_, k)
        shadow_function = eclipse_function_hf(
            self.k,
            array_to_V_hf(u_[:3]),
            array_to_V_hf(u_[3:]),
            r_sec,
            self.R_sec,
            self.R_primary,
            True,
        )

        return shadow_function


class NodeCrossEvent(BaseEvent):
    """Detect equatorial node (ascending or descending) crossings.

    Parameters
    ----------
    terminal: bool, optional
        Whether to terminate integration when the event occurs, defaults to False.
    direction: float, optional
        Handle triggering of event based on whether the node is crossed from above
        i.e. descending node, or is crossed from below i.e. ascending node, defaults to 0,
        i.e. event is triggered during both crossings.

    """

    def __init__(self, terminal=False, direction=0):
        super().__init__(terminal, direction)

    def __call__(self, t, u_, k):
        self._last_t = t
        # Check if the z coordinate of the satellite is zero.
        return u_[2]


class LosEvent(BaseEvent):
    """Detect whether there exists a LOS between two satellites.

    Parameters
    ----------
    attractor: ~hapsira.bodies.body
        The central attractor with respect to which the position vectors of the satellites are defined.
    pos_coords: ~astropy.quantity.Quantity
        A list of position coordinates for the secondary body. These coordinates
        can be found by propagating the body for a desired amount of time.

    """

    def __init__(self, attractor, tofs, secondary_rr, terminal=False, direction=0):
        super().__init__(terminal, direction)
        self._attractor = attractor
        self._secondary_hf = interp_hb(tofs.to_value(u.s), secondary_rr.to_value(u.km))
        self._R = self._attractor.R.to_value(u.km)

    def __call__(self, t, u_, k):
        self._last_t = t

        if norm_V_vf(*u_[:3]) < self._R:
            warn(
                "The norm of the position vector of the primary body is less than the radius of the attractor."
            )

        delta_angle = line_of_sight_gf(  # pylint: disable=E1120
            u_[:3],
            np.array(self._secondary_hf(t)),
            self._R,
        )
        return delta_angle
