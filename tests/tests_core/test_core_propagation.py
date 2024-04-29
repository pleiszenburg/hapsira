from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
import pytest

from hapsira.core.propagation.danby import danby_coe_vf, DANBY_NUMITER, DANBY_RTOL
from hapsira.core.propagation.farnocchia import farnocchia_coe_vf
from hapsira.core.propagation.gooding import (
    gooding_coe_vf,
    GOODING_NUMITER,
    GOODING_RTOL,
)
from hapsira.core.propagation.markley import markley_coe_vf
from hapsira.core.propagation.mikkola import mikkola_coe_vf
from hapsira.core.propagation.pimienta import pimienta_coe_vf
from hapsira.core.propagation.recseries import (
    recseries_coe_vf,
    RECSERIES_METHOD_RTOL,
    RECSERIES_ORDER,
    RECSERIES_NUMITER,
    RECSERIES_RTOL,
)
from hapsira.core.propagation.vallado import vallado_coe_vf, VALLADO_NUMITER
from hapsira.examples import iss


@pytest.mark.parametrize(
    "propagator_coe",
    [
        lambda *args: danby_coe_vf(*args, DANBY_NUMITER, DANBY_RTOL),
        markley_coe_vf,
        pimienta_coe_vf,
        mikkola_coe_vf,
        farnocchia_coe_vf,
        lambda *args: gooding_coe_vf(*args, GOODING_NUMITER, GOODING_RTOL),
        lambda *args: recseries_coe_vf(
            *args,
            RECSERIES_METHOD_RTOL,
            RECSERIES_ORDER,
            RECSERIES_NUMITER,
            RECSERIES_RTOL,
        ),
        lambda *args: vallado_coe_vf(*args, VALLADO_NUMITER),
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
