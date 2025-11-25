from __future__ import annotations
import pyglet
from typing import Optional

from .ui_context import UIContext
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel
from kit_mandelbrot.domain.viewport import screen_to_complex


class CursorCoordsOverlayConfig(BaseModel):
    x_pad: int = 12
    y_pad: int = 12


class CursorCoordsOverlay(UIElement):
    def __init__(self, config: CursorCoordsOverlayConfig) -> None:
        self._deps: Optional[UIContext] = None
        self._x: int = 0
        self._y: int = 0
        self._config = config

        # init UI elements with default values, will be overriddedn later
        self._label = pyglet.text.Label(
            text="", x=0, y=0, anchor_x="left", anchor_y="bottom"
        )

    def mount(self, window: Window, ctx: UIContext) -> None:
        self._deps = ctx
        self._update_config()

    def unmount(self, window: Window) -> None:
        self._deps = None

    def _update_config(self) -> None:
        if self._deps is None:
            return

        self._label.font_size = self._deps.theme.font_main_size
        self._label.font_name = self._deps.theme.font
        self._label.color = self._deps.theme.text_primary

    def on_config_changed(self, section: Optional[BaseModel]) -> None:
        self._update_config()

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        if self._deps is None:
            return

        self._x = x
        self._y = y
        w, h = self._deps.get_size()
        z = screen_to_complex(self._deps.viewport, x, y, w, h)

        self._label.text = f"z={z}"

    def draw(self) -> None:
        if self._deps is None:
            return

        x = self._x + self._config.x_pad
        y = self._y + self._config.y_pad

        self._label.x = int(x)
        self._label.y = int(y)
        self._label.draw()
