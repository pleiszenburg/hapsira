from numpy import (
    float64 as f8,
    float32 as f4,
    float16 as f2,
    finfo,
    nextafter,  # TODO switch to math module
)

from ...settings import settings


__all__ = [
    "EPS",
    "f8",
    "f4",
    "f2",
    "float_",
    "nextafter",
]


if settings["PRECISION"].value == "f8":
    float_ = f8
elif settings["PRECISION"].value == "f4":
    float_ = f4
elif settings["PRECISION"].value == "f2":
    float_ = f2
else:
    raise ValueError("unsupported precision")


EPS = finfo(float_).eps
