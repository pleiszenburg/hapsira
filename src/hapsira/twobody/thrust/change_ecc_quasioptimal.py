"""Quasi optimal eccentricity-only change, with formulas developed by Pollard.

References
----------
* Pollard, J. E. "Simplified Approach for Assessment of Low-Thrust
  Elliptical Orbit Transfers", 1997.

"""
from astropy import units as u

from hapsira.core.jit import array_to_V_hf
from hapsira.core.thrust.change_ecc_quasioptimal import change_ecc_quasioptimal_hb


def change_ecc_quasioptimal(orb_0, ecc_f, f):
    """Guidance law from the model.
    Thrust is aligned with an inertially fixed direction perpendicular to the
    semimajor axis of the orbit.

    Parameters
    ----------
    orb_0 : Orbit
        Initial orbit, containing all the information.
    ecc_f : float
        Final eccentricity.
    f : float
        Magnitude of constant acceleration
    """

    a_d_hf, delta_V, t_f = change_ecc_quasioptimal_hb(
        orb_0.attractor.k.to(u.km**3 / u.s**2).value,  # k
        orb_0.a.to(u.km).value,  # a
        orb_0.ecc.value,  # ecc_0
        ecc_f,
        array_to_V_hf(orb_0.e_vec),  # e_vec,
        array_to_V_hf(orb_0.h_vec),  # h_vec,
        array_to_V_hf(orb_0.r),  # r
        f,
    )

    return a_d_hf, delta_V, t_f
