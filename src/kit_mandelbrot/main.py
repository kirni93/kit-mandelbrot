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


class MandelbrotApp(pyglet.window.Window):
    def __init__(
        self,
        start_re_min=-2.5,
        start_re_max=1.0,
        start_imag_min=-1.5,
        start_imag_max=1.5,
        max_iterations: int = 30,
        start_width: int = 900,
        start_height: int = 600,
    ):
        super().__init__(
            width=start_width,
            height=start_height,
            caption="Mandelbrot Viewer",
            resizable=True,
        )

        # Create ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.viewport = (0, 0, self.width, self.height)

        self.mandelbrot_viewport = Viewport(
            re_min=start_re_min,
            re_max=start_re_max,
            imag_min=start_imag_min,
            imag_max=start_imag_max,
        )
        self.max_iterations = max_iterations

        # Shader to display a single-channel texture
        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_pos;
                in vec2 in_uv;
                out vec2 v_uv;
                void main() {
                    gl_Position = vec4(in_pos, 0.0, 1.0);
                    v_uv = in_uv;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex;
                in vec2 v_uv;
                out vec4 f_color;
                void main() {
                    float s = texture(tex, v_uv).r;      // 0..1 stability
                    f_color = vec4(vec3(s), 1.0);        // grayscale for now
                }
            """,
        )

        # Fullscreen quad
        vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                1.0,
            ],
            dtype="f4",
        )
        indices = np.array([0, 1, 2, 2, 3, 0], dtype="i4")

        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")], self.ibo
        )

        self._debounce_scheduled = False
        self.texture = None

        self.fps_display = pyglet.window.FPSDisplay(window=self)

        self.recompute_texture()  # initial compute using current window size
        self._selecting = False
        self._sel_start_px = (0, 0)
        self._sel_end_px = (0, 0)
        self._sel_batch = pyglet.graphics.Batch()
        self._sel_rect = shapes.Rectangle(
            0, 0, 0, 0, color=(80, 160, 255), batch=self._sel_batch
        )
        self._sel_rect.opacity = 80

    # --- mouse handlers: create/update/commit selection ------------------------
    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT:
            self._selecting = True
            self._sel_start_px = (x, y)
            self._sel_end_px = (x, y)
            self._update_selection_rect()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self._selecting and (buttons & mouse.LEFT):
            self._sel_end_px = (x, y)
            self._update_selection_rect()

    def on_mouse_release(self, x, y, button, modifiers):
        if button == mouse.LEFT and self._selecting:
            self._selecting = False
            # Ignore tiny drags
            if self._sel_rect.width < 4 or self._sel_rect.height < 4:
                self._sel_rect.width = self._sel_rect.height = 0
                return

            w, h = self.get_size()
            (x0, y0), (x1, y1) = self._sel_start_px, self._sel_end_px
            xf0, yf0 = screen_to_fracs(x0, y0, w, h)
            xf1, yf1 = screen_to_fracs(x1, y1, w, h)

            # >>> Apply your new viewport method here:
            self.mandelbrot_viewport.zoom_box(xf0, yf0, xf1, yf1)

            # Clear visual box
            self._sel_rect.width = self._sel_rect.height = 0

            # Recompute now (or debounce if you prefer)
            pyglet.clock.unschedule(self._debounced_recompute)
            pyglet.clock.schedule_once(self._debounced_recompute, 0.0)

    def _update_selection_rect(self):
        (x0, y0), (x1, y1) = self._sel_start_px, self._sel_end_px
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        self._sel_rect.x, self._sel_rect.y = x, y
        self._sel_rect.width, self._sel_rect.height = w, h

    def recompute_texture(self, target_size=None):
        """Compute stability for current (or provided) size and upload as float texture."""
        if target_size is None:
            width, height = self.get_size()
        else:
            width, height = target_size

        c_grid = generate_complex_grid(width, height, self.mandelbrot_viewport)
        # stability = mandelbrot_stability(c_grid, max_iterations=self.max_iterations)
        stability = mandelbrot_stability_vec(
            c_grid=c_grid, max_iterations=self.max_iterations
        )

        tex_data = np.flipud(stability).astype("f4")

        if self.texture is None or self.texture.size != (width, height):
            if self.texture is not None:
                self.texture.release()
            self.texture = self.ctx.texture(
                (width, height), 1, tex_data.tobytes(), dtype="f4"
            )
            self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.texture.write(tex_data.tobytes())

        self.ctx.viewport = (0, 0, self.width, self.height)

    # --- Zoom with mouse wheel ------------------------------------------------
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """
        Natural-feel zoom:
        - scroll_y > 0  => zoom in
        - scroll_y < 0  => zoom out
        Zoom is exponential so small notches feel consistent at all scales.
        """
        # Convert wheel "notches" to scale: 0.9 ^ notches feels nice
        zoom_per_notch = 0.9
        zoom = (zoom_per_notch**scroll_y) if scroll_y != 0 else 1.0

        # Fractional position under the cursor
        w, h = self.get_size()
        xf, yf = screen_to_fracs(x, y, w, h)

        # Update viewport in-place
        self.mandelbrot_viewport.zoom_viewport_at(xf, yf, zoom)

        # Debounced recompute (keeps UI responsive)
        pyglet.clock.unschedule(self._debounced_recompute)
        pyglet.clock.schedule_once(self._debounced_recompute, DEBOUNCE_SEC)

    def on_draw(self):
        self.clear()
        self.ctx.clear(0.07, 0.07, 0.09, 1.0)

        if self.texture is not None:
            self.texture.use(0)
            self.vao.render()

    def _debounced_recompute(self, dt):
        self._debounce_scheduled = False
        self.recompute_texture()

    def on_resize(self, width, height):
        super().on_resize(width, height)
        # Update viewport immediately so the last texture scales smoothly
        self.ctx.viewport = (0, 0, width, height)
        # Debounce the expensive recompute
        if not self._debounce_scheduled:
            self._debounce_scheduled = True
            pyglet.clock.schedule_once(self._debounced_recompute, DEBOUNCE_SEC)
        else:
            # reschedule by clearing and scheduling again
            pyglet.clock.unschedule(self._debounced_recompute)
            pyglet.clock.schedule_once(self._debounced_recompute, DEBOUNCE_SEC)


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

        self._recompute_and_upload(w=width, h=height)

    def _recompute_and_upload(self, w: int, h: int) -> None:
        frame: np.ndarray = self.app.engine.compute(
            width=w, height=h, viewport=self.app.viewport
        )

        self.app.presenter.upload(frame)

    def on_draw(self) -> None:
        self.clear()
        self.app.gl_ctx.clear(0.07, 0.07, 0.09, 1.0)
        self.app.pipeline.draw()

    def on_resize(self, width: int, height: int) -> None:
        super().on_resize(width, height)
        self.app.gl_ctx.viewport = (0, 0, width, height)
        self.app.presenter.ensure_size((width, height))
        self._recompute_and_upload(w=width, h=height)


def main():
    app = MandelbrotWindow()
    pyglet.app.run()

    app.close()


if __name__ == "__main__":
    main()
