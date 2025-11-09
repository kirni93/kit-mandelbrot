import pyglet
from pyglet import shapes
from .Viewport import Viewport


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
