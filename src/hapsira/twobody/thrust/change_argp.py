"""Argument of perigee change, with formulas developed by Pollard.

References
----------
* Pollard, J. E. "Simplified Approach for Assessment of Low-Thrust
  Elliptical Orbit Transfers", 1997.
* Pollard, J. E. "Evaluation of Low-Thrust Orbital Maneuvers", 1998.

"""
from astropy import units as u

from hapsira.core.thrust.change_argp import change_argp_hb


def change_argp(k, a, ecc, argp_0, argp_f, f):
    """Guidance law from the model.
    Thrust is aligned with an inertially fixed direction perpendicular to the
    semimajor axis of the orbit.

    Parameters
    ----------
    k : ~astropy.units.quantity.Quantity
        Gravitational parameter (km**3 / s**2)
    a : ~astropy.units.quantity.Quantity
        Semi-major axis (km)
    ecc : float
        Eccentricity
    argp_0 : ~astropy.units.quantity.Quantity
        Initial argument of periapsis (rad)
    argp_f : ~astropy.units.quantity.Quantity
        Final argument of periapsis (rad)
    f : ~astropy.units.quantity.Quantity
        Magnitude of constant acceleration (km / s**2)

    Returns
    -------
    a_d, delta_V, t_f : tuple (function, ~astropy.units.quantity.Quantity, ~astropy.time.Time)
    """
    a_d_hf, delta_V, t_f = change_argp_hb(
        k=k.to_value(u.km**3 / u.s**2),
        a=a.to_value(u.km),
        ecc=ecc.to_value() if hasattr(ecc, "to_value") else ecc,
        argp_0=argp_0.to_value(u.rad),
        argp_f=argp_f.to_value(u.rad),
        f=f.to_value(u.km / u.s**2),
    )

    return a_d_hf, delta_V, t_f * u.s  # delta_V is scalar, TODO add unit to it?
