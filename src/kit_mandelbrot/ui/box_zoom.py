from __future__ import annotations
from typing import Optional

from pyglet.shapes import BorderedRectangle

from .ui_context import UIContext
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel
from kit_mandelbrot.domain.viewport import screen_to_complex
from pyglet.window import mouse


class BoxZoomConfig(BaseModel):
    opacity: int = 200


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
        if buttons & mouse.LEFT:
            if not self._dragging:
                self._x_start = x
                self._y_start = y
                self._dragging = True

            self._x = x
            self._y = y

    def on_mouse_release(self, x: int, y: int, button, modifiers) -> None:
        if button & mouse.LEFT:
            if self._dragging:
                self._dragging = False
                self._x_end = x
                self._y_end = y

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
