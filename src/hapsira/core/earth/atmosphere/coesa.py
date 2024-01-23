"""This script holds several utilities related to atmospheric computations."""

from ...jit import hjit

__all__ = [
    "get_index_hf",
    "check_altitude_hf",
]


@hjit("f(f,f)")
def _geometric_to_geopotential_hf(z, r0):
    """Converts from given geometric altitude to geopotential one.

    Parameters
    ----------
    z : float
        Geometric altitude.
    r0 : float
        Planet/Natural satellite radius.

    Returns
    -------
    h: float
        Geopotential altitude.
    """
    h = r0 * z / (r0 + z)
    return h


@hjit("f(f,f)")
def _geopotential_to_geometric_hf(h, r0):
    """Converts from given geopotential altitude to geometric one.

    Parameters
    ----------
    h : float
        Geopotential altitude.
    r0 : float
        Planet/Natural satellite radius.

    Returns
    -------
    z: float
        Geometric altitude.
    """
    z = r0 * h / (r0 - h)
    return z


_z_to_h_hf = _geometric_to_geopotential_hf
_h_to_z_hf = _geopotential_to_geometric_hf


@hjit("f(f,f,f)")
def _gravity_hf(z, g0, r0):
    """Relates Earth gravity field magnitude with the geometric height.

    Parameters
    ----------
    z : float
        Geometric height.
    g0 : float
        Gravity value at sea level.
    r0 : float
        Planet/Natural satellite radius.

    Returns
    -------
    g: float
        Gravity value at given geometric altitude.
    """
    g = g0 * (r0 / (r0 + z)) ** 2
    return g


@hjit  # ("i8(f,f)")  # TODO use tuple with fixed length
def get_index_hf(x, x_levels):
    """Finds element in list and returns index.

    Parameters
    ----------
    x : float
        Element to be searched.
    x_levels : list
        List for searching.

    Returns
    -------
    i: int
        Index for the value.

    """
    for i, value in enumerate(x_levels):
        if i < len(x_levels) and value < x:
            continue
        elif x == value:
            return i
        else:
            return i - 1


@hjit("Tuple([f,f])(f,f,b1)")
def check_altitude_hf(alt, r0, geometric):
    # Get geometric and geopotential altitudes
    if geometric:
        z = alt
        h = _z_to_h_hf(z, r0)
    else:
        h = alt
        z = _h_to_z_hf(h, r0)

    return z, h
