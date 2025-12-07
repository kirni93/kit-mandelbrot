from __future__ import annotations
from typing import Optional

from pyglet.shapes import BorderedRectangle

from .ui_context import UIContext
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel
from kit_mandelbrot.domain.viewport import Viewport, screen_to_complex
from pyglet.window import mouse


class BoxZoomConfig(BaseModel):
    opacity: int = 200
    min_box_width: int = 10
    min_box_height: int = 10


class BoxZoom(UIElement):
    def __init__(self, config: BoxZoomConfig) -> None:
        self._deps: Optional[UIContext] = None
        self._x: int = 0
        self._y: int = 0
        self._x_start: int = 0
        self._y_start: int = 0
        self._x_end: int = 0
        self._y_end: int = 0
        self._dragging: bool = False
        self._config = config
        self._box = BorderedRectangle(0, 0, 0, 0, border=1)

    def mount(self, window: Window, ctx: UIContext) -> None:
        self._deps = ctx
        self._update_config()

    def unmount(self, window: Window) -> None:
        self._deps = None

    def _update_config(self) -> None:
        if self._deps is None:
            return

        self._build_components()

    def on_config_changed(self, section: Optional[BaseModel]) -> None:
        self._update_config()

    def _build_components(self) -> None:
        if self._deps is None:
            return

        self._box.color = self._deps.theme.panel_bg
        self._box.border_color = self._deps.theme.panel_border_active
        self._box.opacity = self._config.opacity

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons, modifiers
    ) -> None:
        if not (buttons & mouse.LEFT):
            return

        if not self._dragging:
            self._x_start = x
            self._y_start = y
            self._dragging = True

        self._x = x
        self._y = y

    def _set_viewport(self) -> None:
        if self._deps is None:
            return

        screen_w, screen_h = self._deps.get_size()

        min_x = min(self._x_start, self._x_end)
        max_x = max(self._x_start, self._x_end)
        min_y = min(self._y_start, self._y_end)
        max_y = max(self._y_start, self._y_end)

        if abs(max_x - min_x) < self._config.min_box_width:
            return

        if abs(max_y - min_y) < self._config.min_box_height:
            return

        c1 = screen_to_complex(self._deps.viewport, min_x, min_y, screen_w, screen_h)
        c2 = screen_to_complex(self._deps.viewport, max_x, max_y, screen_w, screen_h)

        re_min = min(c1.real, c2.real)
        re_max = max(c1.real, c2.real)
        imag_min = min(c1.imag, c2.imag)
        imag_max = max(c1.imag, c2.imag)

        self._deps.update_viewport(
            Viewport(
                re_min=re_min,
                re_max=re_max,
                imag_min=imag_min,
                imag_max=imag_max,
            )
        )

    def on_mouse_release(self, x: int, y: int, button, modifiers) -> None:
        if not (button & mouse.LEFT):
            return

        if not self._dragging:
            return

        self._dragging = False
        self._x_end = x
        self._y_end = y

        self._set_viewport()

    def draw(self) -> None:
        if self._deps is None:
            return

        if self._dragging:
            draw_x = min(self._x, self._x_start)
            draw_y = min(self._y, self._y_start)

            w = abs(self._x - self._x_start)
            h = abs(self._y - self._y_start)

            self._box.x = draw_x
            self._box.y = draw_y
            self._box.width = w
            self._box.height = h

            self._box.draw()
