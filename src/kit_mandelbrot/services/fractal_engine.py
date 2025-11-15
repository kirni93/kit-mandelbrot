import moderngl
import numpy as np
from kit_mandelbrot.domain.viewport import Viewport
from typing import Iterator, Optional, Protocol
from kit_mandelbrot.config import ESCPAE_RADIUS
from math import log
from importlib.resources import files
from kit_mandelbrot.rendering.quad import FullscreenQuad
from typing import cast

from kit_mandelbrot.rendering.texture_presenter import TexturePresenter

LN2 = np.log(2.0)


class FractalEngine(Protocol):
    def compute(self, width: int, height: int, viewport: Viewport) -> None: ...


class FractalEngineCPU:
    def compute(self, width: int, height: int, viewport: Viewport) -> None:
        raise NotImplementedError("CPU based calculated disabled for now.")
        # complex_grid = generate_complex_grid(width=width, height=height, vp=viewport)

        # stability_grid = mandelbrot_stability_vec(
        #   c_grid=complex_grid, max_iterations=100, smooth=True
        # )
        # return stability_grid


class FractalEngineGPU:
    def __init__(self, ctx: moderngl.Context, presenter: TexturePresenter) -> None:
        self.ctx = ctx
        self.presenter = presenter

        fs_src = (
            files("kit_mandelbrot.shaders") / "mandelbrot_stability.frag.glsl"
        ).read_text("utf-8")

        vs_src = (files("kit_mandelbrot.shaders") / "present.vert.glsl").read_text(
            "utf-8"
        )

        self.program = ctx.program(vertex_shader=vs_src, fragment_shader=fs_src)

        self.quad = FullscreenQuad(ctx, self.program)

        # Defaults
        self.max_iter = 100
        self.smooth = True

    def compute(self, width: int, height: int, viewport: Viewport) -> None:
        self.presenter.ensure_size((width, height))

        tex = self.presenter.texture

        assert tex is not None

        # create a framebuffer that renders into the R32F texture
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.use()

        # Set the viewport dimensions
        self.ctx.viewport = (0, 0, width, height)

        fbo.clear(0.0, 0.0, 0.0, 0.0)

        # Set uniforms
        cast(moderngl.Uniform, self.program["re_min"]).value = float(viewport.re_min)
        cast(moderngl.Uniform, self.program["re_max"]).value = float(viewport.re_max)

        cast(moderngl.Uniform, self.program["imag_min"]).value = float(
            viewport.imag_min
        )
        cast(moderngl.Uniform, self.program["imag_max"]).value = float(
            viewport.imag_max
        )

        cast(moderngl.Uniform, self.program["max_iter"]).value = int(self.max_iter)
        cast(moderngl.Uniform, self.program["smooth_stability"]).value = int(
            self.smooth
        )

        self.quad.draw()
        self.ctx.screen.use()


def z_generator(c: complex) -> Iterator[complex]:
    """
    z_0 = 0
    z_n+1 = z_n^2 + c
    """
    z: complex = 0j

    while True:
        yield z
        z = z**2 + c


def escape_time(
    c: complex, max_iterations: int, smooth: bool = False
) -> Optional[float]:
    for n, z in enumerate(z_generator(c)):
        z_abs = abs(z)

        if z_abs > ESCPAE_RADIUS:
            if smooth:
                return n + 1 - log(log(z_abs)) / log(2)
            return n

        if n > max_iterations:
            return None


def mandelbrot_stability(c_grid: np.ndarray, max_iterations: int) -> np.ndarray:
    """
    Return maks to specify if a point in the c_grid is part of the mandelbrot set.
    """
    height, width = c_grid.shape

    # fill all with zeroes
    stability = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            c = complex(c_grid[y, x])
            n = escape_time(c, max_iterations)
            stability[y, x] = 1.0 if n is None else (n / max_iterations)

    return stability


def generate_complex_grid(width: int, height: int, vp: Viewport) -> np.ndarray:
    """Height and Width array of complex numbers for the viewport."""
    re = np.linspace(start=vp.re_min, stop=vp.re_max, num=width, dtype=np.float32)

    imag = np.linspace(
        start=vp.imag_min, stop=vp.imag_max, num=height, dtype=np.float32
    )

    Re, Im = np.meshgrid(re, imag)
    return Re + 1j * Im


def mandelbrot_stability_vec(
    c_grid: np.ndarray,
    max_iterations: int,
    smooth: bool = False,
    escape_radius: float = 2.0,
) -> np.ndarray:
    """
    Vectorized Mandelbrot stability:
    - returns 1.0 for points that never escape within max_iterations
    - else returns (n / max_iterations) or smooth-normalized if smooth=True
    """
    # Ensure complex dtype for math
    c = c_grid.astype(np.complex128, copy=False)

    # Working arrays
    z = np.zeros_like(c)
    escaped = np.zeros(c.shape, dtype=bool)  # which pixels have escaped (ever)
    n_at_escape = np.zeros(
        c.shape, dtype=np.float32
    )  # store n (or smooth n) when they escape

    esc2 = escape_radius * escape_radius

    for n in range(1, max_iterations + 1):
        # Iterate all (computing everywhere is often faster than masked writes)
        z = z * z + c

        # Check magnitude^2 to avoid sqrt
        mag2 = z.real * z.real + z.imag * z.imag

        # Pixels that just escaped this iteration
        newly = (~escaped) & (mag2 > esc2)
        if newly.any():
            if smooth:
                # Smooth escape time: n + 1 - log(log|z|)/log 2
                # log|z| = 0.5 * log(|z|^2) => log(log|z|) = log(0.5 * log mag2)
                # Use safe logs; mag2 > esc2 >= 4, so log is well-defined
                mu = n + 1.0 - (np.log(np.log(np.sqrt(mag2[newly]))) / LN2)
                n_at_escape[newly] = mu
            else:
                n_at_escape[newly] = n

            escaped[newly] = True

        # Early out if everyone escaped
        if escaped.all():
            break

    # Map to stability in [0, 1]:
    # - points that never escaped → 1.0
    # - escaped → normalized by max_iterations
    stability = np.empty(c.shape, dtype=np.float32)
    stability[~escaped] = 1.0
    if smooth:
        # Normalize smooth mu by max_iterations, clamp to [0, 1]
        s = n_at_escape / max_iterations
        np.clip(s, 0.0, 1.0, out=s)
        stability[escaped] = s[escaped]
    else:
        stability[escaped] = n_at_escape[escaped] / max_iterations

    return stability
