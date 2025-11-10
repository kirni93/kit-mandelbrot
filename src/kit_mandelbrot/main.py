import numpy as np
import plotly.express as px
import pyglet
from kit_mandelbrot.config import DEBOUNCE_SEC
from pyglet import shapes
from pyglet.window import mouse
import moderngl
from kit_mandelbrot.domain import viewport
from kit_mandelbrot.domain.viewport import Viewport
from kit_mandelbrot.services.fractal_engine import (
    FractalEngine,
    FractalEngineCPU,
    generate_complex_grid,
    mandelbrot_stability_vec,
)
from importlib.resources import files
from kit_mandelbrot.rendering.texture_presenter import TexturePresenter
from kit_mandelbrot.rendering.quad import FullscreenQuad
from kit_mandelbrot.rendering.pipeline import RenderPipeline
from kit_mandelbrot.app_context import AppContext
from kit_mandelbrot.ui.cursor_coords import (
    CursorCoordsOverlay,
    CursorCoordsOverlayConfig,
)
from kit_mandelbrot.ui.dependencies import UIDeps
from kit_mandelbrot.ui.manager import UIManager


def plot_mandelbrot(
    stability: np.ndarray, vp: Viewport, width: int, height: int
) -> None:
    fig = px.imshow(
        stability,
        origin="lower",
        zmin=0.0,
        zmax=1.0,
        x=np.linspace(vp.re_min, vp.re_max, width),
        y=np.linspace(vp.imag_min, vp.imag_max, height),
        color_continuous_scale=[
            (0.0, "midnightblue"),
            (0.5, "white"),
            (0.65, "yellow"),
            (0.8, "red"),
            (1.0, "black"),  # inside the set
        ],
    )

    fig.update_layout(
        xaxis_title="Re(c)",
        yaxis_title="Im(c)",
    )

    fig.show()


# --- helpers: viewport math ---------------------------------------------------
def screen_to_fracs(x: int, y: int, width: int, height: int) -> tuple[float, float]:
    """Convert screen px to [0..1] fractions, origin at bottom-left (pyglet)."""
    return (x / max(1, width), y / max(1, height))


start_re_min = -2.5
start_re_max = 1.0
start_imag_min = -1.5
start_imag_max = 1.5


class MandelbrotWindow(pyglet.window.Window):
    def __init__(self, width: int = 900, height: int = 600) -> None:
        super().__init__(
            width=width, height=height, caption="Mandelbrot Viewer", resizable=True
        )
        ctx = moderngl.create_context()
        ctx.viewport = (0, 0, self.width, self.height)

        vs = (files("kit_mandelbrot.shaders") / "present.vert.glsl").read_text("utf-8")
        fs = (files("kit_mandelbrot.shaders") / "present.frag.glsl").read_text("utf-8")
        program = ctx.program(vertex_shader=vs, fragment_shader=fs)

        presenter = TexturePresenter(ctx)
        presenter.ensure_size((self.width, self.height))  # allocate texture

        engine: FractalEngine = FractalEngineCPU()

        quad = FullscreenQuad(ctx, program)
        pipeline = RenderPipeline(ctx, program, quad, presenter)
        vp = Viewport(
            re_min=start_re_min,
            re_max=start_re_max,
            imag_min=start_imag_min,
            imag_max=start_imag_max,
        )

        self.app = AppContext(
            gl_ctx=ctx,
            presenter=presenter,
            pipeline=pipeline,
            engine=engine,
            viewport=vp,
        )

        deps = UIDeps(get_size=self.get_size, viewport=vp)

        self._recompute_and_upload(w=width, h=height)

        self.set_mouse_visible(True)

        cursor = self.get_system_mouse_cursor(self.CURSOR_CROSSHAIR)
        self.set_mouse_cursor(cursor)

        self.ui = UIManager(window=self, deps=deps)

    def _recompute_and_upload(self, w: int, h: int) -> None:
        frame: np.ndarray = self.app.engine.compute(
            width=w, height=h, viewport=self.app.viewport
        )

        self.app.presenter.upload(frame)

    def on_draw(self) -> None:
        self.clear()
        self.app.gl_ctx.clear(0.07, 0.07, 0.09, 1.0)
        self.app.pipeline.draw()

        self.ui.draw()

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.app.gl_ctx.viewport = (0, 0, width, height)
        self.app.presenter.ensure_size((width, height))
        self._recompute_and_upload(w=width, h=height)


def main():
    app = MandelbrotWindow()

    cursor_cords_config = CursorCoordsOverlayConfig()
    cursor_cords = CursorCoordsOverlay(cursor_cords_config)

    app.ui.add(cursor_cords)

    pyglet.app.run()

    app.close()


if __name__ == "__main__":
    main()
