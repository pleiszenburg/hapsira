from astropy.io import ascii as ascii_
from astropy.utils.data import get_pkg_data_filename

from numpy import float32 as f4

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
]

# Following constants come from original U.S Atmosphere 1962 paper so a pure
# model of this atmosphere can be implemented
R = f4(8314.32)  # u.J / u.kmol / u.K
R_air = f4(287.053)  # u.J / u.kg / u.K
k = f4(1.380622e-23)  # u.J / u.K
Na = f4(6.022169e-26)  # 1 / u.kmol
g0 = f4(9.80665)  # u.m / u.s**2
r0 = f4(6356.766)  # u.km
M0 = f4(28.9644)  # u.kg / u.kmol
P0 = f4(101325)  # u.Pa
T0 = f4(288.15)  # u.K
Tinf = 1000  # u.K
gamma = f4(1.4)  # one
alpha = f4(34.1632)  # u.K / u.km
beta = f4(1.458e-6)  # (u.kg / u.s / u.m / (u.K) ** 0.5)
S = f4(110.4)  # u.K

# Reading layer parameters file
coesa76_data = ascii_.read(get_pkg_data_filename("data/coesa76.dat"))
b_levels = tuple(f4(number) for number in coesa76_data["b"].data)
zb_levels = tuple(f4(number) for number in coesa76_data["Zb [km]"].data)  # u.km
hb_levels = tuple(f4(number) for number in coesa76_data["Hb [km]"].data)  # u.km
Tb_levels = tuple(f4(number) for number in coesa76_data["Tb [K]"].data)  # u.K
Lb_levels = tuple(f4(number) for number in coesa76_data["Lb [K/km]"].data)  # u.K / u.km
pb_levels = tuple(f4(number) for number in coesa76_data["pb [mbar]"].data)  # u.mbar

# Reading pressure and density coefficients files
p_data = ascii_.read(get_pkg_data_filename("data/coesa76_p.dat"))
rho_data = ascii_.read(get_pkg_data_filename("data/coesa76_rho.dat"))

# Zip coefficients for each altitude
z_coeff = tuple(f4(number) for number in p_data["z [km]"].data)  # u.km
p_coeff = (
    tuple(f4(number) for number in p_data["A"].data),
    tuple(f4(number) for number in p_data["B"].data),
    tuple(f4(number) for number in p_data["C"].data),
    tuple(f4(number) for number in p_data["D"].data),
    tuple(f4(number) for number in p_data["E"].data),
)
rho_coeff = (
    tuple(f4(number) for number in rho_data["A"].data),
    tuple(f4(number) for number in rho_data["B"].data),
    tuple(f4(number) for number in rho_data["C"].data),
    tuple(f4(number) for number in rho_data["D"].data),
    tuple(f4(number) for number in rho_data["E"].data),
)
