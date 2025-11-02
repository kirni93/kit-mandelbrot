from typing import Iterator, Optional
from typing import Iterator, Final
from dataclasses import dataclass
import numpy as np
import plotly.express as px

ESCPAE_RADIUS: float = 2.0
ESCPAE_RADIUS2: float = ESCPAE_RADIUS * ESCPAE_RADIUS


@dataclass(frozen=True)
class Viewport:
    re_min: float
    re_max: float
    imag_min: float
    imag_max: float


def z_generator(c: complex) -> Iterator[complex]:
    """
    z_0 = 0
    z_n+1 = z_n^2 + c
    """
    z: complex = 0j

    while True:
        yield z
        z = z**2 + c


def generate_complex_grid(width: int, height: int, vp: Viewport) -> np.ndarray:
    """Height and Width array of complex numbers for the viewport."""
    re = np.linspace(start=vp.re_min, stop=vp.re_max, num=width, dtype=np.float64)

    imag = np.linspace(
        start=vp.imag_min, stop=vp.imag_max, num=height, dtype=np.float64
    )

    Re, Im = np.meshgrid(re, imag)
    return Re + 1j * Im


def escape_time(c: complex, max_iterations: int) -> Optional[int]:
    for n, z in enumerate(z_generator(c)):
        if abs(z) > ESCPAE_RADIUS:
            return n

        if n > max_iterations:
            return None


def mandelbrot_mask(c_grid: np.ndarray, max_iterations: int) -> np.ndarray:
    """
    Return maks to specify if a point in the c_grid is part of the mandelbrot set.
    """
    height, width = c_grid.shape

    # fill all with zeroes
    mask = np.zeros((height, width), dtype=bool)

    for y in range(height):
        for x in range(width):
            c = complex(c_grid[y, x])

            mask[y, x] = escape_time(c, max_iterations) is None

    return mask


def main():
    MAX_ITERATIONS = 100

    width = 900
    height = 600

    vp = Viewport(re_min=-2.5, re_max=1.0, imag_min=-1.5, imag_max=1.5)

    main_grid = generate_complex_grid(width, height, vp)

    mask = mandelbrot_mask(c_grid=main_grid, max_iterations=MAX_ITERATIONS)

    mask_int = mask.astype(np.int8)

    fig = px.imshow(mask_int, color_continuous_scale=["black", "white"], origin="lower")

    fig.show()


if __name__ == "__main__":
    main()
