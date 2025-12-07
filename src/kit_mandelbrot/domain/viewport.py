from typing import NamedTuple


class Viewport(NamedTuple):
    re_min: float
    re_max: float
    imag_min: float
    imag_max: float


def from_points(c1: complex, c2: complex) -> Viewport:
    re_min = min(c1.real, c2.real)
    re_max = max(c1.real, c2.real)
    imag_min = min(c1.imag, c2.imag)
    imag_max = max(c1.imag, c2.imag)

    return Viewport(
        re_min=re_min,
        re_max=re_max,
        imag_min=imag_min,
        imag_max=imag_max,
    )


def screen_to_complex(vp: Viewport, x: int, y: int, width: int, height: int) -> complex:
    """Convert screen coordinates to a complex-plane coordinate"""
    re = vp.re_min + (x / width) * (vp.re_max - vp.re_min)
    imag = vp.imag_min + (y / height) * (vp.imag_max - vp.imag_min)

    return complex(real=re, imag=imag)
