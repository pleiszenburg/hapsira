"""Earth focused orbital mechanics routines."""


from astropy import units as u

from hapsira.bodies import Earth
from hapsira.core.jit import hjit, djit
from hapsira.core.math.linalg import add_VV_hf
from hapsira.core.perturbations import J2_perturbation_hf
from hapsira.core.propagation.base import func_twobody_hf
from hapsira.earth.enums import EarthGravity
from hapsira.twobody.propagation import CowellPropagator


class EarthSatellite:
    """Position and velocity of a body with respect to Earth
    at a given time.
    """

    def __init__(self, orbit, spacecraft):
        """Constructor.

        Parameters
        ----------
        orbit : Orbit
            Position and velocity of a body with respect to an attractor
            at a given time (epoch).
        spacecraft : Spacecraft

        Raises
        ------
        ValueError
            If the orbit's attractor is not Earth

        """
        if orbit.attractor is not Earth:
            raise ValueError("The attractor must be Earth")

        self._orbit = orbit  # type: Orbit
        self._spacecraft = spacecraft  # type: Spacecraft

    @property
    def orbit(self):
        """Orbit of the EarthSatellite."""
        return self._orbit

    @property
    def spacecraft(self):
        """Spacecraft of the EarthSatellite."""
        return self._spacecraft

    @u.quantity_input(tof=u.min)
    def propagate(self, tof, atmosphere=None, gravity=None, *args):
        """Propagates an 'EarthSatellite Orbit' at a specified time.

        If value is true anomaly, propagate orbit to this anomaly and return the result.
        Otherwise, if time is provided, propagate this `EarthSatellite Orbit` some `time` and return the result.

        Parameters
        ----------
        tof : ~astropy.units.Quantity, ~astropy.time.Time, ~astropy.time.TimeDelta
            Scalar time to propagate.
        atmosphere:
            a callable model from hapsira.earth.atmosphere
        gravity : EarthGravity
            There are two possible values, SPHERICAL and J2. Only J2 is implemented at the moment. Default value is None.
        *args:
            parameters used in perturbation models.

        Returns
        -------
        EarthSatellite
            A new EarthSatellite with the propagated Orbit

        """

        if gravity not in (None, EarthGravity.J2):
            raise NotImplementedError

        if atmosphere is not None:
            # Cannot compute density without knowing the state,
            # the perturbations parameters are not always fixed
            raise NotImplementedError

        if gravity:
            J2_ = Earth.J2.value
            R_ = Earth.R.to_value(u.km)

            @hjit("V(f,V,V,f)", cache=False)
            def ad_hf(t0, rr, vv, k):
                return J2_perturbation_hf(t0, rr, vv, k, J2_, R_)

        else:

            @hjit("V(f,V,V,f)")
            def ad_hf(t0, rr, vv, k):
                return 0.0, 0.0, 0.0

        @djit(cache=False)
        def f_hf(t0, rr, vv, k):
            du_kep_rr, du_kep_vv = func_twobody_hf(t0, rr, vv, k)
            du_ad_vv = ad_hf(t0, rr, vv, k)
            return du_kep_rr, add_VV_hf(du_kep_vv, du_ad_vv)

        new_orbit = self.orbit.propagate(tof, method=CowellPropagator(f=f_hf))
        return EarthSatellite(new_orbit, self.spacecraft)
