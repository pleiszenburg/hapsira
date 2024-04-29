"""The U.S. Standard Atmosphere 1976 is an idealized, steady-state model of
mean annual conditions of Earth's atmosphere from the surface to 1000 km at
latitude 45N, as it is assumed to exist during a period with moderate solar
activity. The defining meteorological elements are sea-level temperature and
pressure, and a temperature-height profile to 1000 km. The air is assumed to be
dry, and at heights sufficiently below 86 km, the atmosphere is assumed to be
homogeneously mixed with a relative-volume composition leading to a constant
mean molecular weight.

Since 1976 many constants such us Earth's radius or Avogadro's number have been
updated. In order to have a pure COESA76 atmospheric model, the official paper
values were used.

+--------+---------+---------+-----------+---------------+---------------+
| Z (km) |  H (km) |  T (K)  |  p (mbar) | rho (kg / m3) | beta (K / km) |
+--------+---------+---------+-----------+---------------+---------------+
|   0.0  |   0.0   | 288.150 | 1.01325e3 |     1.2250    |      -6.5     |
+--------+---------+---------+-----------+---------------+---------------+
| 11.019 |   11.0  | 216.650 |  2.2632e2 |   3.6392e-1   |      0.0      |
+--------+---------+---------+-----------+---------------+---------------+
| 20.063 |   20.0  | 216.650 |  5.4748e1 |   8.8035e-2   |      1.0      |
+--------+---------+---------+-----------+---------------+---------------+
| 32.162 |   32.0  | 228.650 |  8.6801e0 |   1.3225e-2   |      2.8      |
+--------+---------+---------+-----------+---------------+---------------+
| 47.350 |   47.0  | 270.650 |  1.1090e0 |   1.4275e-3   |      0.0      |
+--------+---------+---------+-----------+---------------+---------------+
| 51.413 |   51.0  | 270.650 | 6.6938e-1 |   8.6160e-4   |      -2.8     |
+--------+---------+---------+-----------+---------------+---------------+
| 71.802 |   71.0  | 214.650 | 3.9564e-2 |   6.4211e-5   |      -2.0     |
+--------+---------+---------+-----------+---------------+---------------+
|  86.0  | 84.8520 |  186.87 | 3.7338e-3 |    6.958e-6   |      0.0      |
+--------+---------+---------+-----------+---------------+---------------+
|  91.0  |  89.716 |  186.87 | 1.5381e-3 |    2.860e-6   |   elliptical  |
+--------+---------+---------+-----------+---------------+---------------+
|  110.0 | 108.129 |  240.00 | 7.1042e-5 |    9.708e-8   |      12.0     |
+--------+---------+---------+-----------+---------------+---------------+
|  120.0 | 117.777 |  360.00 | 2.5382e-5 |    2.222e-8   |  exponential  |
+--------+---------+---------+-----------+---------------+---------------+
|  500.0 | 463.540 |  999.24 | 3.0236e-9 |   5.215e-13   |  exponential  |
+--------+---------+---------+-----------+---------------+---------------+
| 1000.0 | 864.071 |   1000  | 7.5138e-5 |   3.561e-15   |  exponential  |
+--------+---------+---------+-----------+---------------+---------------+

"""

from astropy import units as u
import numpy as np

from hapsira.earth.atmosphere.base import COESA

from hapsira.core.earth.atmosphere.coesa76 import (
    R,
    R_air,
    k,
    Na,
    g0,
    r0,
    M0,
    P0,
    T0,
    Tinf,
    gamma,
    alpha,
    beta,
    S,
    b_levels,
    zb_levels,
    hb_levels,
    Tb_levels,
    Lb_levels,
    pb_levels,
    z_coeff,
    p_coeff,
    rho_coeff,
    pressure_vf,
    density_vf,
    temperature_vf,
)

__all__ = [
    "COESA76",
]

R = R * u.J / u.kmol / u.K
R_air = R_air * u.J / u.kg / u.K
k = k * u.J / u.K
Na = Na / u.kmol
g0 = g0 * u.m / u.s**2
r0 = r0 * u.km
M0 = M0 * u.kg / u.kmol
P0 = P0 * u.Pa
T0 = T0 * u.K
Tinf = Tinf * u.K
alpha = alpha * u.K / u.km
beta = beta * (u.kg / u.s / u.m / (u.K) ** 0.5)
S = S * u.K

# Reading layer parameters file
b_levels = np.array(b_levels)
zb_levels = np.array(zb_levels) * u.km
hb_levels = np.array(hb_levels) * u.km
Tb_levels = np.array(Tb_levels) * u.K
Lb_levels = np.array(Lb_levels) * u.K / u.km
pb_levels = np.array(pb_levels) * u.mbar

# Zip coefficients for each altitude
z_coeff = z_coeff * u.km
p_coeff = [np.array(entry) for entry in p_coeff]
rho_coeff = [np.array(entry) for entry in rho_coeff]


class COESA76(COESA):
    """Holds the model for U.S Standard Atmosphere 1976."""

    def __init__(self):
        """Constructor for the class."""
        super().__init__(
            b_levels, zb_levels, hb_levels, Tb_levels, Lb_levels, pb_levels
        )

    def _get_coefficients_avobe_86(self, z, table_coeff):
        """Returns corresponding coefficients for 4th order polynomial approximation.

        Parameters
        ----------
        z : ~astropy.units.Quantity
            Geometric altitude
        table_coeff : list
            List containing different coefficient lists.

        Returns
        -------
        coeff_list: list
            List of corresponding coefficients
        """
        # Get corresponding coefficients
        i = self._get_index(z, z_coeff)
        coeff_list = []
        for X_set in table_coeff:
            coeff_list.append(X_set[i])

        return coeff_list

    def temperature(self, alt, geometric=True):
        """Solves for temperature at given altitude.

        Parameters
        ----------
        alt : ~astropy.units.Quantity
            Geometric/Geopotential altitude.
        geometric : bool
            If `True`, assumes geometric altitude kind.

        Returns
        -------
        T: ~astropy.units.Quantity
            Kinetic temeperature.
        """

        return temperature_vf(alt.to_value(u.km), geometric) * u.K

    def pressure(self, alt, geometric=True):
        """Solves pressure at given altitude.

        Parameters
        ----------
        alt : ~astropy.units.Quantity
            Geometric/Geopotential altitude.
        geometric : bool
            If `True`, assumes geometric altitude kind.

        Returns
        -------
        p: ~astropy.units.Quantity
            Pressure at given altitude.
        """

        return pressure_vf(alt.to_value(u.km), geometric) * u.Pa

    def density(self, alt, geometric=True):
        """Solves density at given height.

        Parameters
        ----------
        alt : ~astropy.units.Quantity
            Geometric/Geopotential height.
        geometric : bool
            If `True`, assumes that `alt` argument is geometric kind.

        Returns
        -------
        rho: ~astropy.units.Quantity
            Density at given height.
        """

        return density_vf(alt.to_value(u.km), geometric) * u.kg / u.m**3

    def properties(self, alt, geometric=True):
        """Solves temperature, pressure, density at given height.

        Parameters
        ----------
        alt : ~astropy.units.Quantity
            Geometric/Geopotential height.
        geometric : bool
            If `True`, assumes that `alt` argument is geometric kind.

        Returns
        -------
        T: ~astropy.units.Quantity
            Temperature at given height.
        p: ~astropy.units.Quantity
            Pressure at given height.
        rho: ~astropy.units.Quantity
            Density at given height.
        """
        T = self.temperature(alt, geometric=geometric)
        p = self.pressure(alt, geometric=geometric)
        rho = self.density(alt, geometric=geometric)

        return T, p, rho

    def sound_speed(self, alt, geometric=True):
        """Solves speed of sound at given height.

        Parameters
        ----------
        alt : ~astropy.units.Quantity
            Geometric/Geopotential height.
        geometric : bool
            If `True`, assumes that `alt` argument is geometric kind.

        Returns
        -------
        Cs: ~astropy.units.Quantity
            Speed of Sound at given height.
        """
        # Check if valid range and convert to geopotential
        z, h = self._check_altitude(alt, r0, geometric=geometric)

        if z > 86 * u.km:
            raise ValueError(
                "Speed of sound in COESA76 has just been implemented up to 86km."
            )
        T = self.temperature(alt, geometric).value
        # Using eqn-(50)
        Cs = ((gamma * R_air.value * T) ** 0.5) * (u.m / u.s)

        return Cs

    def viscosity(self, alt, geometric=True):
        """Solves dynamic viscosity at given height.

        Parameters
        ----------
        alt : ~astropy.units.Quantity
            Geometric/Geopotential height.
        geometric : bool
            If `True`, assumes that `alt` argument is geometric kind.

        Returns
        -------
        mu: ~astropy.units.Quantity
            Dynamic viscosity at given height.
        """
        # Check if valid range and convert to geopotential
        z, h = self._check_altitude(alt, r0, geometric=geometric)

        if z > 86 * u.km:
            raise ValueError(
                "Dynamic Viscosity in COESA76 has just been implemented up to 86km."
            )
        T = self.temperature(alt, geometric).value
        # Using eqn-(51)
        mu = (beta.value * T**1.5 / (T + S.value)) * (u.N * u.s / (u.m) ** 2)

        return mu

    def thermal_conductivity(self, alt, geometric=True):
        """Solves coefficient of thermal conductivity at given height.

        Parameters
        ----------
        alt : ~astropy.units.Quantity
            Geometric/Geopotential height.
        geometric : bool
            If `True`, assumes that `alt` argument is geometric kind.

        Returns
        -------
        k: ~astropy.units.Quantity
            coefficient of thermal conductivity at given height.
        """
        # Check if valid range and convert to geopotential
        z, h = self._check_altitude(alt, r0, geometric=geometric)

        if z > 86 * u.km:
            raise ValueError(
                "Thermal conductivity in COESA76 has just been implemented up to 86km."
            )
        T = self.temperature(alt, geometric=geometric).value
        # Using eqn-(53)
        k = (2.64638e-3 * T**1.5 / (T + 245.4 * (10 ** (-12.0 / T)))) * (
            u.J / u.m / u.s / u.K
        )

        return k
