from abc import ABC, abstractmethod
from math import degrees as rad2deg
from typing import Callable

from astropy import units as u
from astropy.coordinates import get_body_barycentric_posvel

from hapsira.core.jit import hjit
from hapsira.core.math.ivp import dense_interp_brentq_hb
from hapsira.core.math.linalg import mul_Vs_hf, norm_V_hf
from hapsira.core.events import (
    eclipse_function_hf,
    line_of_sight_hf,
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

    @abstractmethod
    def __init__(self, terminal, direction):
        self._terminal, self._direction = terminal, direction
        self._last_t = None
        self._impl_hf = None
        self._impl_dense_hf = None

    @property
    def terminal(self):
        return self._terminal

    @property
    def direction(self):
        return self._direction

    @property
    def last_t(self):
        return self._last_t << u.s

    @property
    def last_t_raw(self) -> float:
        return self._last_t

    @last_t_raw.setter
    def last_t_raw(self, value: float):
        self._last_t = value

    @property
    def impl_hf(self) -> Callable:
        return self._impl_hf

    @property
    def impl_dense_hf(self) -> Callable:
        return self._impl_dense_hf

    def _wrap(self):
        self._impl_dense_hf = dense_interp_brentq_hb(self._impl_hf)


class AltitudeCrossEvent(BaseEvent):
    """Detect if a satellite crosses a specific threshold altitude.

    Parameters
    ----------
    alt: float
        Threshold altitude from the ground (km).
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

        @hjit("f(f,V,V,f)", cache=False)
        def impl_hf(t, rr, vv, k):
            r_norm = norm_V_hf(rr)
            return (
                r_norm - R - alt
            )  # If this goes from +ve to -ve, altitude is decreasing.

        self._impl_hf = impl_hf
        self._wrap()


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

        R = orbit.attractor.R.to_value(u.m)
        R_polar = orbit.attractor.R_polar.to_value(u.m)
        lat = lat.to_value(u.deg)  # Threshold latitude (in degrees).

        @hjit("f(f,V,V,f)", cache=False)
        def impl_hf(t, rr, vv, k):
            pos_on_body = mul_Vs_hf(rr, R / norm_V_hf(rr))
            _, lat_, _ = cartesian_to_ellipsoidal_hf(R, R_polar, *pos_on_body)
            return rad2deg(lat_) - lat

        self._impl_hf = impl_hf
        self._wrap()


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
        primary_body = orbit.attractor
        secondary_body = orbit.attractor.parent
        epoch = orbit.epoch

        self._R_sec = secondary_body.R.to_value(u.km)
        self._R_primary = primary_body.R.to_value(u.km)

        epochs = time_range(start=epoch, end=epoch + tof, num_values=steps)
        r_primary_wrt_ssb, _ = get_body_barycentric_posvel(primary_body.name, epochs)
        r_secondary_wrt_ssb, _ = get_body_barycentric_posvel(
            secondary_body.name, epochs
        )
        self._r_sec_hf = interp_hb(
            (epochs - epoch).to_value(u.s),
            (r_secondary_wrt_ssb - r_primary_wrt_ssb).xyz.to_value(u.km),
        )


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

    def __init__(self, orbit, tof, steps=50, terminal=False, direction=0):
        super().__init__(orbit, tof, steps, terminal, direction)

        R_sec = self._R_sec
        R_primary = self._R_primary
        r_sec_hf = self._r_sec_hf

        @hjit("f(f,V,V,f)", cache=False)
        def impl_hf(t, rr, vv, k):
            r_sec = r_sec_hf(t)
            shadow_function = eclipse_function_hf(
                k,
                rr,
                vv,
                r_sec,
                R_sec,
                R_primary,
                False,
            )
            return shadow_function

        self._impl_hf = impl_hf
        self._wrap()


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

    def __init__(self, orbit, tof, steps=50, terminal=False, direction=0):
        super().__init__(orbit, tof, steps, terminal, direction)

        R_sec = self._R_sec
        R_primary = self._R_primary
        r_sec_hf = self._r_sec_hf

        @hjit("f(f,V,V,f)", cache=False)
        def impl_hf(t, rr, vv, k):
            r_sec = r_sec_hf(t)
            shadow_function = eclipse_function_hf(
                k,
                rr,
                vv,
                r_sec,
                R_sec,
                R_primary,
                True,
            )
            return shadow_function

        self._impl_hf = impl_hf
        self._wrap()


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

        @hjit("f(f,V,V,f)", cache=False)
        def impl_hf(t, rr, vv, k):
            # Check if the z coordinate of the satellite is zero.
            return rr[2]

        self._impl_hf = impl_hf
        self._wrap()


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
        secondary_hf = interp_hb(tofs.to_value(u.s), secondary_rr.to_value(u.km))
        R = attractor.R.to_value(u.km)

        @hjit("f(f,V,V,f)", cache=False)
        def impl_hf(t, rr, vv, k):
            # Can currently not warn due to: https://github.com/numba/numba/issues/1243
            # TODO Matching test deactivated ...
            # if norm_V_hf(rr) < R:
            #     warn(
            #         "The norm of the position vector of the primary body is less than the radius of the attractor."
            #     )
            delta_angle = line_of_sight_hf(
                rr,
                secondary_hf(t),
                R,
            )
            return delta_angle

        self._impl_hf = impl_hf
        self._wrap()
