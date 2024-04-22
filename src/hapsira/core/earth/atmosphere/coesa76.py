from math import exp

from astropy.io import ascii as ascii_
from astropy.utils.data import get_pkg_data_filename

from numpy import float32 as float_, int64 as i8

from .coesa import check_altitude_hf
from ...jit import hjit, vjit

__all__ = [
    "R",
    "R_air",
    "k",
    "Na",
    "g0",
    "r0",
    "M0",
    "P0",
    "T0",
    "Tinf",
    "gamma",
    "alpha",
    "beta",
    "S",
    "b_levels",
    "zb_levels",
    "hb_levels",
    "Tb_levels",
    "Lb_levels",
    "pb_levels",
    "z_coeff",
    "p_coeff",
    "rho_coeff",
    "COESA76_GEOMETRIC",
    "pressure_hf",
    "pressure_vf",
    "temperature_hf",
    "temperature_vf",
    "density_hf",
    "density_vf",
]

# Following constants come from original U.S Atmosphere 1962 paper so a pure
# model of this atmosphere can be implemented
R = float_(8314.32)  # u.J / u.kmol / u.K
R_air = float_(287.053)  # u.J / u.kg / u.K
k = float_(1.380622e-23)  # u.J / u.K
Na = float_(6.022169e-26)  # 1 / u.kmol
g0 = float_(9.80665)  # u.m / u.s**2
r0 = float_(6356.766)  # u.km
M0 = float_(28.9644)  # u.kg / u.kmol
P0 = float_(101325)  # u.Pa
T0 = float_(288.15)  # u.K
Tinf = 1000  # u.K
gamma = float_(1.4)  # one
alpha = float_(34.1632)  # u.K / u.km
beta = float_(1.458e-6)  # (u.kg / u.s / u.m / (u.K) ** 0.5)
S = float_(110.4)  # u.K

# Reading layer parameters file
coesa76_data = ascii_.read(get_pkg_data_filename("data/coesa76.dat"))
b_levels = tuple(i8(number) for number in coesa76_data["b"].data)
zb_levels = tuple(float_(number) for number in coesa76_data["Zb [km]"].data)  # u.km
hb_levels = tuple(float_(number) for number in coesa76_data["Hb [km]"].data)  # u.km
Tb_levels = tuple(float_(number) for number in coesa76_data["Tb [K]"].data)  # u.K
Lb_levels = tuple(
    float_(number) for number in coesa76_data["Lb [K/km]"].data
)  # u.K / u.km
pb_levels = tuple(float_(number) for number in coesa76_data["pb [mbar]"].data)  # u.mbar

# Reading pressure and density coefficients files
p_data = ascii_.read(get_pkg_data_filename("data/coesa76_p.dat"))
rho_data = ascii_.read(get_pkg_data_filename("data/coesa76_rho.dat"))

# Zip coefficients for each altitude
z_coeff = tuple(i8(number) for number in p_data["z [km]"].data)  # u.km
p_coeff = (
    tuple(float_(number) for number in p_data["A"].data),
    tuple(float_(number) for number in p_data["B"].data),
    tuple(float_(number) for number in p_data["C"].data),
    tuple(float_(number) for number in p_data["D"].data),
    tuple(float_(number) for number in p_data["E"].data),
)
rho_coeff = (
    tuple(float_(number) for number in rho_data["A"].data),
    tuple(float_(number) for number in rho_data["B"].data),
    tuple(float_(number) for number in rho_data["C"].data),
    tuple(float_(number) for number in rho_data["D"].data),
    tuple(float_(number) for number in rho_data["E"].data),
)

COESA76_GEOMETRIC = True


@hjit("Tuple([f,f])(f,f,b1)")
def _check_altitude_hf(alt, r0, geometric):
    """Checks if altitude is inside valid range.

    Parameters
    ----------
    alt : float
        Altitude to be checked.
    r0 : float
        Attractor radius.
    geometric : bool
        If `True`, assumes geometric altitude kind.
        Default `True`.

    Returns
    -------
    z : float
        Geometric altitude.
    h : float
        Geopotential altitude.

    """
    z, h = check_altitude_hf(alt, r0, geometric)
    assert zb_levels[0] <= z <= zb_levels[-1]

    return z, h


@hjit("i8(f)")
def _get_index_zb_levels_hf(x):
    """Finds element in list and returns index.

    Parameters
    ----------
    x : float
        Element to be searched.

    Returns
    -------
    i : int
        Index for the value. `999` if there was an error.

    """
    for i, value in enumerate(zb_levels):
        if i < len(zb_levels) and value < x:
            continue
        if x == value:
            return i
        return i - 1
    return 999  # HACK error ... ?


@hjit("i8(f)")
def _get_index_z_coeff_hf(x):
    """Finds element in list and returns index.

    Parameters
    ----------
    x : float
        Element to be searched.

    Returns
    -------
    i : int
        Index for the value.
        Index for the value. `999` if there was an error.

    """
    for i, value in enumerate(z_coeff):
        if i < len(z_coeff) and value < x:
            continue
        if x == value:
            return i
        return i - 1
    return 999  # HACK error ... ?


@hjit("Tuple([f,f,f,f,f])(f)")
def _get_coefficients_avobe_86_p_coeff_hf(z):
    """Returns corresponding coefficients for 4th order polynomial approximation.

    Parameters
    ----------
    z : float
        Geometric altitude

    Returns
    -------
    coeffs : tuple[float,float,float,float,float]
        Tuple of corresponding coefficients

    """
    # Get corresponding coefficients
    i = _get_index_z_coeff_hf(z)
    return p_coeff[0][i], p_coeff[1][i], p_coeff[2][i], p_coeff[3][i], p_coeff[4][i]


@hjit("Tuple([f,f,f,f,f])(f)")
def _get_coefficients_avobe_86_rho_coeff_hf(z):
    """Returns corresponding coefficients for 4th order polynomial approximation.

    Parameters
    ----------
    z : float
        Geometric altitude

    Returns
    -------
    coeffs : tuple[float,float,float,float,float]
        Tuple of corresponding coefficients

    """
    # Get corresponding coefficients
    i = _get_index_z_coeff_hf(z)
    return (
        rho_coeff[0][i],
        rho_coeff[1][i],
        rho_coeff[2][i],
        rho_coeff[3][i],
        rho_coeff[4][i],
    )


@hjit("f(f,b1)")
def temperature_hf(alt, geometric):
    """Solves for temperature at given altitude.

    Parameters
    ----------
    alt : float
        Geometric/Geopotential altitude.
    geometric : bool
        If `True`, assumes geometric altitude kind.
        Default `True`.

    Returns
    -------
    T : float
        Kinetic temeperature.

    """
    # Test if altitude is inside valid range
    z, h = _check_altitude_hf(alt, r0, geometric)

    # Get base parameters
    i = _get_index_zb_levels_hf(z)
    Tb = Tb_levels[i]
    Lb = Lb_levels[i]
    hb = hb_levels[i]

    # Apply different equations
    if z < zb_levels[7]:
        # Below 86km
        # TODO: Apply air mean molecular weight ratio factor
        Tm = Tb + Lb * (h - hb)
        T = Tm
    elif zb_levels[7] <= z and z < zb_levels[8]:
        # [86km, 91km)
        T = 186.87
    elif zb_levels[8] <= z and z < zb_levels[9]:
        # [91km, 110km]
        Tc = 263.1905
        A = -76.3232
        a = -19.9429
        T = Tc + A * (1 - ((z - zb_levels[8]) / a) ** 2) ** 0.5
    elif zb_levels[9] <= z and z < zb_levels[10]:
        # [110km, 120km]
        T = 240 + Lb * (z - zb_levels[9])
    else:
        T10 = 360.0
        _gamma = Lb_levels[9] / (Tinf - T10)
        epsilon = (z - zb_levels[10]) * (r0 + zb_levels[10]) / (r0 + z)
        T = Tinf - (Tinf - T10) * exp(-_gamma * epsilon)

    return T


@vjit("f(f,b1)")
def temperature_vf(alt, geometric):
    """Solves for temperature at given altitude.
    Vectorized.

    Parameters
    ----------
    alt : float
        Geometric/Geopotential altitude.
    geometric : bool
        If `True`, assumes geometric altitude kind.
        Default `True`.

    Returns
    -------
    T : float
        Kinetic temeperature.

    """
    return temperature_hf(alt, geometric)


@hjit("f(f,b1)")
def pressure_hf(alt, geometric):
    """Solves pressure at given altitude.

    Parameters
    ----------
    alt : float
        Geometric/Geopotential altitude.
    geometric : bool
        If `True`, assumes geometric altitude kind.
        Default `True`.

    Returns
    -------
    p : float
        Pressure at given altitude.

    """
    # Test if altitude is inside valid range
    z, h = _check_altitude_hf(alt, r0, geometric)

    # Obtain gravity magnitude
    # Get base parameters
    i = _get_index_zb_levels_hf(z)
    Tb = Tb_levels[i]
    Lb = Lb_levels[i]
    hb = hb_levels[i]
    pb = pb_levels[i]

    # If above 86[km] usual formulation is applied
    if z < 86:
        if Lb == 0.0:
            p = pb * exp(-alpha * (h - hb) / Tb) * 100  # HACK 100 ... SI-prefix change?
        else:
            T = temperature_hf(z, geometric)
            p = pb * (Tb / T) ** (alpha / Lb) * 100  # HACK 100 ... SI-prefix change?
    else:
        # TODO: equation (33c) should be applied instead of using coefficients

        # A 4th order polynomial is used to approximate pressure.  This was
        # directly taken from: http://www.braeunig.us/space/atmmodel.htm
        A, B, C, D, E = _get_coefficients_avobe_86_p_coeff_hf(z)

        # Solve the polynomial
        p = exp(A * z**4 + B * z**3 + C * z**2 + D * z + E)

    return p


@vjit("f(f,b1)")
def pressure_vf(alt, geometric):
    """Solves pressure at given altitude.
    Vectorized.

    Parameters
    ----------
    alt : float
        Geometric/Geopotential altitude.
    geometric : bool
        If `True`, assumes geometric altitude kind.
        Default `True`.

    Returns
    -------
    p : float
        Pressure at given altitude.

    """
    return pressure_hf(alt, geometric)


@hjit("f(f,b1)")
def density_hf(alt, geometric):
    """Solves density at given height.

    Parameters
    ----------
    alt : float
        Geometric/Geopotential height.
    geometric : bool
        If `True`, assumes that `alt` argument is geometric kind.
        Default `True`.

    Returns
    -------
    rho : float
        Density at given height.

    """
    # Test if altitude is inside valid range
    z, _ = _check_altitude_hf(alt, r0, geometric)

    # Solve temperature and pressure
    if z <= 86:
        T = temperature_hf(z, geometric)
        p = pressure_hf(z, geometric)
        rho = p / R_air / T
    else:
        # TODO: equation (42) should be applied instead of using coefficients

        # A 4th order polynomial is used to approximate pressure.  This was
        # directly taken from: http://www.braeunig.us/space/atmmodel.htm
        A, B, C, D, E = _get_coefficients_avobe_86_rho_coeff_hf(z)

        # Solve the polynomial
        rho = exp(A * z**4 + B * z**3 + C * z**2 + D * z + E)

    return rho


@vjit("f(f,b1)")
def density_vf(alt, geometric):
    """Solves density at given height.
    Vectorized.

    Parameters
    ----------
    alt : float
        Geometric/Geopotential height.
    geometric : bool
        If `True`, assumes that `alt` argument is geometric kind.
        Default `True`.

    Returns
    -------
    rho : float
        Density at given height.

    """
    return density_hf(alt, geometric)
