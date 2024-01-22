from ..jit import djit


__all__ = [
    "func_twobody_hf",
]


@djit
def func_twobody_hf(t0, rr, vv, k):
    """Differential equation for the initial value two body problem.

    Parameters
    ----------
    t0 : float
        Time.
    rr : tuple[float,float,float]
        Position vector
    vv : tuple[float,float,float]
        Velocity vector.
    k : float
        Standard gravitational parameter.

    """
    x, y, z = rr
    vx, vy, vz = vv
    r3 = (x**2 + y**2 + z**2) ** 1.5

    return (vx, vy, vz), (-k * x / r3, -k * y / r3, -k * z / r3)
