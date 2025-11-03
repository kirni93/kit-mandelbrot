from typing import Iterator, Optional
from dataclasses import dataclass
import numpy as np
import plotly.express as px
from math import log
import pyglet
from pyglet import shapes
from pyglet.window import mouse
import moderngl

ESCPAE_RADIUS: float = 2.0
ESCPAE_RADIUS2: float = ESCPAE_RADIUS * ESCPAE_RADIUS
DEBOUNCE_SEC = 0.15
LN2 = np.log(2.0)


@dataclass
class Viewport:
    re_min: float
    re_max: float
    imag_min: float
    imag_max: float

    # --- zoom ---------------------------------------------------------------
    def zoom_viewport_at(self, x_frac: float, y_frac: float, zoom: float) -> None:
        """
        Zoom viewport around a fractional screen position (x_frac, y_frac).
        zoom < 1 => zoom in, zoom > 1 => zoom out. Modifies self in-place.
        Keeps the mouse position fixed in view.
        """
        re_span = self.re_max - self.re_min
        im_span = self.imag_max - self.imag_min

        # current complex coordinates under cursor
        re_cursor = self.re_min + x_frac * re_span
        im_cursor = self.imag_min + y_frac * im_span

        new_re_span = re_span * zoom
        new_im_span = im_span * zoom

        # place cursor at same fractional position after zoom
        self.re_min = re_cursor - x_frac * new_re_span
        self.re_max = self.re_min + new_re_span
        self.imag_min = im_cursor - y_frac * new_im_span
        self.imag_max = self.imag_min + new_im_span

    # --- zoom into a box selected on screen ---------------------------------
    def zoom_box(self, xf0: float, yf0: float, xf1: float, yf1: float) -> None:
        """
        Zoom viewport to a box defined by two fractional corners (xf0,yf0) and (xf1,yf1).
        Fractions are 0..1 in both axes, bottom-left origin.
        """
        re_min, re_max = self.re_min, self.re_max
        im_min, im_max = self.imag_min, self.imag_max
        re_span = re_max - re_min
        im_span = im_max - im_min

        # Compute new boundaries in fractal coordinates
        self.re_min = min(xf0, xf1) * re_span + re_min
        self.re_max = max(xf0, xf1) * re_span + re_min
        self.imag_min = min(yf0, yf1) * im_span + im_min
        self.imag_max = max(yf0, yf1) * im_span + im_min

    # --- panning ------------------------------------------------------------
    def pan_by_frac(self, dx_frac: float, dy_frac: float) -> None:
        """
        Translate the viewport by dx_frac, dy_frac of its span.
        Positive dx moves right, positive dy moves up (screen coords: bottom-left).
        """
        re_span = self.re_max - self.re_min
        im_span = self.imag_max - self.imag_min
        self.re_min += dx_frac * re_span
        self.re_max += dx_frac * re_span
        self.imag_min += dy_frac * im_span
        self.imag_max += dy_frac * im_span

    # --- aspect utilities ----------------------------------------------------
    def get_spans(self):
        """Return (re_span, im_span)."""
        return (self.re_max - self.re_min, self.imag_max - self.imag_min)

    def set_imag_span_from_aspect(self, aspect: float) -> None:
        """
        Adjust imag_min/max to match a given aspect ratio (width/height).
        Useful when resetting.
        """
        re_span = self.re_max - self.re_min
        im_span = re_span / aspect
        im_center = (self.imag_min + self.imag_max) / 2.0
        self.imag_min = im_center - im_span / 2.0
        self.imag_max = im_center + im_span / 2.0


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
        c.shape, dtype=np.float64
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
    stability = np.empty(c.shape, dtype=np.float64)
    stability[~escaped] = 1.0
    if smooth:
        # Normalize smooth mu by max_iterations, clamp to [0, 1]
        s = n_at_escape / max_iterations
        np.clip(s, 0.0, 1.0, out=s)
        stability[escaped] = s[escaped]
    else:
        stability[escaped] = n_at_escape[escaped] / max_iterations

    return stability


def mandelbrot_stability(c_grid: np.ndarray, max_iterations: int) -> np.ndarray:
    """
    Return maks to specify if a point in the c_grid is part of the mandelbrot set.
    """
    height, width = c_grid.shape

    # fill all with zeroes
    stability = np.zeros((height, width), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            c = complex(c_grid[y, x])

            n = escape_time(c, max_iterations)
            stability[y, x] = 1.0 if n is None else (n / max_iterations)

    return stability


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


class HUD:
    def __init__(self, pos=(10, 10), padding=8):
        self.batch = pyglet.graphics.Batch()
        self.pos = pos
        self.padding = padding

        # Background panel (auto-resized in update)
        self.bg = shapes.Rectangle(
            x=pos[0], y=pos[1], width=320, height=110, color=(0, 0, 0), batch=self.batch
        )
        self.bg.opacity = 160  # semi-transparent

        # Labels (positions updated in update())
        self.labels = [
            pyglet.text.Label(
                "", x=0, y=0, color=(255, 255, 255, 255), batch=self.batch
            ),
            pyglet.text.Label(
                "", x=0, y=0, color=(200, 200, 200, 255), batch=self.batch
            ),
            pyglet.text.Label(
                "", x=0, y=0, color=(200, 200, 200, 255), batch=self.batch
            ),
            pyglet.text.Label(
                "", x=0, y=0, color=(200, 200, 200, 255), batch=self.batch
            ),
            pyglet.text.Label(
                "", x=0, y=0, color=(200, 200, 200, 255), batch=self.batch
            ),
        ]

    def update(self, viewport: Viewport, iterations, size):
        w, h = size
        re_span = viewport.re_max - viewport.re_min
        im_span = viewport.imag_max - viewport.imag_min
        re_center = (viewport.re_min + viewport.re_max) / 2.0
        im_center = (viewport.imag_min + viewport.imag_max) / 2.0

        # Update label text
        self.labels[0].text = "Viewport"
        self.labels[1].text = f" Center: Re={re_center:.6f}, Im={im_center:.6f}"
        self.labels[2].text = f" Span:   ΔRe={re_span:.6e}, ΔIm={im_span:.6e}"
        self.labels[3].text = f" Iter:   {iterations}    Size: {w}×{h}"

        # Layout labels (top-left anchor of panel)
        x0, y0 = self.pos
        line_h = 18
        for i, lbl in enumerate(self.labels):
            lbl.x = x0 + self.padding
            lbl.y = y0 + self.padding + i * line_h

        # Resize background to fit text block
        max_w = max(lbl.content_width for lbl in self.labels) if self.labels else 0
        total_h = self.padding * 2 + line_h * len(self.labels)
        self.bg.x = x0
        self.bg.y = y0
        self.bg.width = max(200, self.padding * 2 + max_w)
        self.bg.height = total_h

    def draw(self):
        self.batch.draw()


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

        self.hud = HUD(pos=(10, 10))

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

        # Update & draw HUD last
        self.hud.update(
            viewport=self.mandelbrot_viewport,
            iterations=self.max_iterations,
            size=self.get_size(),
        )
        self.hud.draw()
        # self.fps_display.draw()

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


def main():
    app = MandelbrotApp()
    pyglet.app.run()


if __name__ == "__main__":
    main()
