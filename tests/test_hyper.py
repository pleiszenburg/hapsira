import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import special

from hapsira.core.math.special import hyp2f1b_vf


@pytest.mark.parametrize("x", np.linspace(0, 1, num=11))
def test_hyp2f1_battin_scalar(x):
    expected_res = special.hyp2f1(3, 1, 5 / 2, x)

    res = hyp2f1b_vf(x)
    assert_allclose(res, expected_res)
