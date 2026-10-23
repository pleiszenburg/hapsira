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



@pytest.mark.parametrize(
    "propagator_coe",
    [
        danby_coe,
        markley_coe,
        pimienta_coe,
        mikkola_coe,
        gooding_coe,
    ],
)
@pytest.mark.parametrize("ecc", [0.01, 0.3, 0.7])
@pytest.mark.parametrize("nu", [0.3, -0.3, 2.5, -2.5])
@pytest.mark.parametrize("periods", [0.01, 0.71, 2.71, 10.71, -0.01, -2.71])
def test_propagate_with_coe_agrees_with_farnocchia(propagator_coe, ecc, nu, periods):
    # Covers both halves of the orbit, more than one revolution and backward propagation
    k = iss.attractor.k.to_value(u.km**3 / u.s**2)
    a = 10000.0  # km
    p = a * (1 - ecc**2)
    inc, raan, argp = 0.9, 0.5, 1.2
    tof = periods * 2 * np.pi * np.sqrt(a**3 / k)

    nu_final = propagator_coe(k, p, ecc, inc, raan, argp, nu, tof)
    nu_expected = farnocchia_coe(k, p, ecc, inc, raan, argp, nu, tof)

    assert np.isfinite(nu_final)
    # Compare angles modulo a full turn
    assert abs((nu_final - nu_expected + np.pi) % (2 * np.pi) - np.pi) < 1e-7
