from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
import numpy as np
import pytest

from hapsira.core.propagation import (
    danby_coe,
    gooding_coe,
    markley_coe,
    mikkola_coe,
    pimienta_coe,
)
from hapsira.core.propagation.farnocchia import farnocchia_coe
from hapsira.examples import iss


@pytest.mark.parametrize(
    "propagator_coe",
    [
        danby_coe,
        markley_coe,
        pimienta_coe,
        mikkola_coe,
        farnocchia_coe,
        gooding_coe,
    ],
)
def test_propagate_with_coe(propagator_coe):
    period = iss.period
    a, ecc, inc, raan, argp, nu = iss.classical()
    p = a * (1 - ecc**2)

    # Delete the units
    p = p.to_value(u.km)
    ecc = ecc.value
    period = period.to_value(u.s)
    inc = inc.to_value(u.rad)
    raan = raan.to_value(u.rad)
    argp = argp.to_value(u.rad)
    nu = nu.to_value(u.rad)
    k = iss.attractor.k.to_value(u.km**3 / u.s**2)

    nu_final = propagator_coe(k, p, ecc, inc, raan, argp, nu, period)

    assert_quantity_allclose(nu_final, nu)


def _angle_difference(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


@pytest.mark.parametrize("ecc", [1.2, 2.0, 5.0])
@pytest.mark.parametrize("nu", [-1.0, -0.3, 0.3, 1.0])
@pytest.mark.parametrize("tof_units", [0.3, 1.0, 3.0, 10.0, -0.3, -3.0])
def test_danby_hyperbolic_agrees_with_farnocchia(ecc, nu, tof_units):
    # The hyperbolic mean anomaly is not periodic: covers before and after periapsis
    k = iss.attractor.k.to_value(u.km**3 / u.s**2)
    rp = 6800.0  # km
    p = rp * (1 + ecc)
    inc, raan, argp = 0.9, 0.5, 1.2
    tof = tof_units * np.sqrt(rp**3 / k)

    nu_final = danby_coe(k, p, ecc, inc, raan, argp, nu, tof)
    nu_expected = farnocchia_coe(k, p, ecc, inc, raan, argp, nu, tof)

    assert _angle_difference(nu_final, nu_expected) < 1e-7


@pytest.mark.parametrize("ecc", [0.3, 0.9, 0.99, 0.999])
@pytest.mark.parametrize("tof_units", [0.3, 1.0, 3.0, 10.0])
def test_gooding_converges_near_parabolic(ecc, tof_units):
    # The default tolerance must give the documented accuracy, including near e = 1
    k = iss.attractor.k.to_value(u.km**3 / u.s**2)
    rp = 6800.0  # km
    p = rp * (1 + ecc)
    inc, raan, argp, nu = 0.9, 0.5, 1.2, -1.0
    tof = tof_units * np.sqrt(rp**3 / k)

    nu_final = gooding_coe(k, p, ecc, inc, raan, argp, nu, tof)
    nu_expected = farnocchia_coe(k, p, ecc, inc, raan, argp, nu, tof)

    assert _angle_difference(nu_final, nu_expected) < 1e-7
