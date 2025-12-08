from math import fabs
import pyglet
import moderngl
from kit_mandelbrot.domain.viewport import Viewport
from kit_mandelbrot.services.cmd_engine import CommandEngine
from kit_mandelbrot.services.commands.quit import Q_CMD
from kit_mandelbrot.services.commands.toggle_smooth import TOOGLE_SMOOTH_CMD
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
from kit_mandelbrot.ui.box_zoom import BoxZoom, BoxZoomConfig
from kit_mandelbrot.ui.ui_context import UIContext
from kit_mandelbrot.ui.manager import UIManager
from kit_mandelbrot.ui.theme import DEFAULT_THEME
from kit_mandelbrot.ui.cmd_prompt_overlay import (
    CmdPromptOverlay,
    CmdPromptOverlayConfig,
)

from typing import cast


class MandelbrotWindow(pyglet.window.Window):
    def __init__(self, width: int = 900, height: int = 600) -> None:
        super().__init__(
            width=width, height=height, caption="Mandelbrot Viewer", resizable=True
        )

        self._smooth: bool = True
        self._max_iter = 100

        self.push_handlers(pyglet.window.event.WindowEventLogger())

        ctx = moderngl.create_context()
        ctx.viewport = (0, 0, self.width, self.height)

        presenter = TexturePresenter(ctx)
        presenter.ensure_size((self.width, self.height))  # allocate texture

        assert presenter.texture is not None
        engine: FractalEngine = FractalEngineGPU(ctx=ctx, presenter=presenter)

        pipeline = RenderPipeline(
            ctx, presenter=presenter, max_iter=self._max_iter, smooth=self._smooth
        )
        cmd_engine = CommandEngine()
        vp = Viewport()
        self.ui_context = UIContext(
            get_size=self.get_size,
            viewport=vp,
            theme=DEFAULT_THEME,
            update_viewport=self.update_viewport,
            execute_command=cmd_engine.execute,
            get_command=cmd_engine.get_command,
            prompt_suggest=cmd_engine.prompt_suggest,
        )

        self.app = AppContext(
            gl_ctx=ctx,
            presenter=presenter,
            pipeline=pipeline,
            engine=engine,
            update_viewport=self.update_viewport,
            quit=pyglet.app.exit,
            toggle_smooth=self._toggle_smooth,
        )

        self._recompute_and_upload(w=width, h=height)

        self.set_mouse_visible(True)

        cursor = self.get_system_mouse_cursor(self.CURSOR_CROSSHAIR)
        self.set_mouse_cursor(cursor)

        cmd_engine.mount(self.app)
        cmd_engine.register(VP_CMD)
        cmd_engine.register(Q_CMD)
        cmd_engine.register(TOOGLE_SMOOTH_CMD)

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

    def _toggle_smooth(self) -> bool:
        self._smooth = not self._smooth

        self.app.pipeline.set_smooth(self._smooth)

        return self._smooth

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.app.gl_ctx.viewport = (0, 0, width, height)
        self.app.presenter.ensure_size((width, height))
        self._recompute_and_upload(w=width, h=height)


def main():
    app = MandelbrotWindow()

    # cursor_cords_config = CursorCoordsOverlayConfig()
    # cursor_cords = CursorCoordsOverlay(cursor_cords_config)

    # app.ui.add(cursor_cords)

    # viewport_overlay_config = ViewportOverlayConfig()
    # viewport_overlay = ViewportOverlay(viewport_overlay_config)

    # disable until i bother to make it look good
    # app.ui.add(viewport_overlay)

    cmd_prompt_config = CmdPromptOverlayConfig()
    cmd_prompt_overlay = CmdPromptOverlay(cmd_prompt_config)

    app.ui.add(cmd_prompt_overlay)

    box_zoom_config = BoxZoomConfig()
    box_zoom = BoxZoom(box_zoom_config)

    app.ui.add(box_zoom)

    pyglet.app.run()

    app.close()


if __name__ == "__main__":
    main()
