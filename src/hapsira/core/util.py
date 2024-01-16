from math import cos, sin

from numba import njit as jit
import numpy as np

from .jit import hjit, gjit


__all__ = [
    "rotation_matrix_hf",
    "rotation_matrix_gf",
    "alinspace",
    "spherical_to_cartesian",
    "planetocentric_to_AltAz_hf",
]


@hjit("M(f,i8)")
def rotation_matrix_hf(angle, axis):
    c = cos(angle)
    s = sin(angle)
    if axis == 0:
        return (
            (1.0, 0.0, 0.0),
            (0.0, c, -s),
            (0.0, s, c),
        )
    if axis == 1:
        return (
            (c, 0.0, s),
            (0.0, 1.0, 0.0),
            (-s, 0.0, c),
        )
    if axis == 2:
        return (
            (c, -s, 0.0),
            (s, c, 0.0),
            (0.0, 0.0, 1.0),
        )
    raise ValueError("Invalid axis: must be one of 0, 1 or 2")


@gjit("void(f,i8,u1[:],f[:,:])", "(),(),(n)->(n,n)")
def rotation_matrix_gf(angle, axis, dummy, r):
    """
    Vectorized rotation_matrix

    `dummy` because of https://github.com/numba/numba/issues/2797
    """
    assert dummy.shape == (3,)
    (
        (r[0, 0], r[0, 1], r[0, 2]),
        (r[1, 0], r[1, 1], r[1, 2]),
        (r[2, 0], r[2, 1], r[2, 2]),
    ) = rotation_matrix_hf(angle, axis)


@jit
def alinspace(start, stop=None, num=50, endpoint=True):
    """Return increasing, evenly spaced angular values over a specified interval."""
    # Create a new variable to avoid numba crash,
    # See https://github.com/numba/numba/issues/5661
    if stop is None:
        stop_ = start + 2 * np.pi
    elif stop <= start:
        stop_ = stop + (np.floor((start - stop) / 2 / np.pi) + 1) * 2 * np.pi
    else:
        stop_ = stop

    if endpoint:
        return np.linspace(start, stop_, num)
    else:
        return np.linspace(start, stop_, num + 1)[:-1]


@jit
def spherical_to_cartesian(v):
    r"""Compute cartesian coordinates from spherical coordinates (norm, colat, long). This function is vectorized.

    .. math::

       v = norm \cdot \begin{bmatrix}
       \sin(colat)\cos(long)\\
       \sin(colat)\sin(long)\\
       \cos(colat)\\
       \end{bmatrix}

    Parameters
    ----------
    v : numpy.ndarray
        Spherical coordinates in 3D (norm, colat, long). Angles must be in radians.

    Returns
    -------
    v : numpy.ndarray
        Cartesian coordinates (x,y,z)

    """
    v = np.asarray(v)
    norm_vecs = np.expand_dims(np.asarray(v[..., 0]), -1)
    vsin = np.sin(v[..., 1:3])
    vcos = np.cos(v[..., 1:3])
    x = np.asarray(vsin[..., 0] * vcos[..., 1])
    y = np.asarray(vsin[..., 0] * vsin[..., 1])
    z = np.asarray(vcos[..., 0])
    return norm_vecs * np.stack((x, y, z), axis=-1)


@hjit("M(f,f)")
def planetocentric_to_AltAz_hf(theta, phi):
    r"""Defines transformation matrix to convert from Planetocentric coordinate system
    to the Altitude-Azimuth system.

    .. math::
       t\_matrix = \begin{bmatrix}
       -\sin(theta) & \cos(theta) & 0\\
       -\sin(phi)\cdot\cos(theta) & -\sin(phi)\cdot\sin(theta) & \cos(phi)\\
       \cos(phi)\cdot\cos(theta) & \cos(phi)\cdot\sin(theta) & \sin(phi)
       \end{bmatrix}

    Parameters
    ----------
    theta: float
        Local sidereal time
    phi: float
        Planetodetic latitude

    Returns
    -------
    t_matrix: tuple[tuple[float,float,float],...]
        Transformation matrix
    """
    # Transformation matrix for converting planetocentric equatorial coordinates to topocentric horizon system.
    st = sin(theta)
    ct = cos(theta)
    sp = sin(phi)
    cp = cos(phi)

    return (
        (-st, ct, 0.0),
        (
            -sp * ct,
            -sp * st,
            cp,
        ),
        (
            cp * ct,
            cp * st,
            sp,
        ),
    )
