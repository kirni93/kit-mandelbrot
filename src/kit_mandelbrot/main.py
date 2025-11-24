import plotly.express as px
import numpy as np
import pyglet
import moderngl
from kit_mandelbrot.domain.viewport import Viewport
from kit_mandelbrot.services.cmd_engine import CommandEngine
from kit_mandelbrot.services.commands.quit import Q_CMD
from kit_mandelbrot.services.commands.viewport import VP_CMD
from kit_mandelbrot.services.fractal_engine import (
    FractalEngine,
    FractalEngineGPU,
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
from kit_mandelbrot.ui.ui_context import UIContext
from kit_mandelbrot.ui.manager import UIManager
from kit_mandelbrot.ui.theme import DEFAULT_THEME
from kit_mandelbrot.ui.viewport_overlay import ViewportOverlay, ViewportOverlayConfig
from kit_mandelbrot.ui.cmd_prompt_overlay import (
    CmdPromptOverlay,
    CmdPromptOverlayConfig,
)


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


START_RE_MIN = -2.5
START_RE_MAX = 1.0
START_IMAG_MIN = -1.5
START_IMAG_MAX = 1.5


class MandelbrotWindow(pyglet.window.Window):
    def __init__(self, width: int = 900, height: int = 600) -> None:
        super().__init__(
            width=width, height=height, caption="Mandelbrot Viewer", resizable=True
        )

        # DEBUG psuh all window events
        self.push_handlers(pyglet.window.event.WindowEventLogger())

        ctx = moderngl.create_context()
        ctx.viewport = (0, 0, self.width, self.height)

        vs = (files("kit_mandelbrot.shaders") / "present.vert.glsl").read_text("utf-8")
        fs = (files("kit_mandelbrot.shaders") / "present_color.frag.glsl").read_text(
            "utf-8"
        )
        program = ctx.program(vertex_shader=vs, fragment_shader=fs)

        presenter = TexturePresenter(ctx)
        presenter.ensure_size((self.width, self.height))  # allocate texture

        # engine: FractalEngine = FractalEngineCPU()
        assert presenter.texture is not None
        engine: FractalEngine = FractalEngineGPU(ctx=ctx, presenter=presenter)

        quad = FullscreenQuad(ctx, program)
        pipeline = RenderPipeline(ctx, program, quad, presenter)
        cmd_engine = CommandEngine()
        vp = Viewport(
            re_min=START_RE_MIN,
            re_max=START_RE_MAX,
            imag_min=START_IMAG_MIN,
            imag_max=START_IMAG_MAX,
        )
        self.ui_context = UIContext(
            get_size=self.get_size,
            viewport=vp,
            theme=DEFAULT_THEME,
            update_viewport=self.update_viewport,
            execute_command=cmd_engine.execute,
        )

        self.app = AppContext(
            gl_ctx=ctx,
            presenter=presenter,
            pipeline=pipeline,
            engine=engine,
            update_viewport=self.update_viewport,
            quit=pyglet.app.exit,
        )

        self._recompute_and_upload(w=width, h=height)

        self.set_mouse_visible(True)

        cursor = self.get_system_mouse_cursor(self.CURSOR_CROSSHAIR)
        self.set_mouse_cursor(cursor)

        cmd_engine.mount(self.app)
        cmd_engine.register(VP_CMD)
        cmd_engine.register(Q_CMD)

        self.ui = UIManager(window=self, ctx=self.ui_context)

    def update_viewport(self, vp: Viewport) -> None:
        self.ui_context.viewport = vp
        self._recompute_and_upload(w=self.width, h=self.height)

    def on_key_press(
        self, symbol: int, modifiers: int
    ) -> pyglet.event.EVENT_HANDLE_STATE:
        if symbol == pyglet.window.key.ESCAPE:
            return pyglet.event.EVENT_HANDLED

        return super().on_key_press(symbol, modifiers)

    def _recompute_and_upload(self, w: int, h: int) -> None:
        self.app.engine.compute(width=w, height=h, viewport=self.ui_context.viewport)

    def on_draw(self) -> None:
        self.clear()
        self.app.gl_ctx.clear(0.07, 0.07, 0.09, 1.0)
        self.app.pipeline.draw()

        self.ui.draw()

    def on_resize(self, width: int, height: int) -> None:
        self.app.gl_ctx.viewport = (0, 0, width, height)
        self.app.presenter.ensure_size((width, height))
        self._recompute_and_upload(w=width, h=height)


def main():
    app = MandelbrotWindow()

    cursor_cords_config = CursorCoordsOverlayConfig()
    cursor_cords = CursorCoordsOverlay(cursor_cords_config)

    app.ui.add(cursor_cords)

    viewport_overlay_config = ViewportOverlayConfig()
    viewport_overlay = ViewportOverlay(viewport_overlay_config)

    app.ui.add(viewport_overlay)

    cmd_prompt_config = CmdPromptOverlayConfig()
    cmd_prompt_overlay = CmdPromptOverlay(cmd_prompt_config)

    app.ui.add(cmd_prompt_overlay)

    pyglet.app.run()

    app.close()


if __name__ == "__main__":
    main()
