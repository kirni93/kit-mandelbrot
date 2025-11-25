from typing import NamedTuple


class Viewport(NamedTuple):
    re_min: float
    re_max: float
    imag_min: float
    imag_max: float


def screen_to_complex(vp: Viewport, x: int, y: int, width: int, height: int) -> complex:
    """Convert screen coordinates to a complex-plane coordinate"""
    re = vp.re_min + (x / width) * (vp.re_max - vp.re_min)
    imag = vp.imag_max - (y / height) * (vp.imag_max - vp.imag_min)

    return complex(real=re, imag=imag)
