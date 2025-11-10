from __future__ import annotations

import pyglet
from typing import Optional
from .dependencies import UIDeps
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel


class CursorCoordsOverlayConfig(BaseModel):
    x_pad: int = 12
    y_pad: int = 12
    font_size: int = 12
    font_name: str = "Menlo"
    color: tuple[int, int, int, int] = (230, 230, 230, 255)


class CursorCoordsOverlay(UIElement):
    def __init__(self, config: CursorCoordsOverlayConfig) -> None:
        self._deps: Optional[UIDeps] = None
        self._x: int = 0
        self._y: int = 0
        self._config = config

        # init UI elements with default values, will be overriddedn later
        self._label = pyglet.text.Label(
            text="", x=0, y=0, anchor_x="left", anchor_y="bottom"
        )
        self._update_config()

    def mount(self, window: Window, deps: UIDeps) -> None:
        self._deps = deps

    def unmount(self, window: Window) -> None:
        self._deps = None

    def _update_config(self) -> None:
        self._label.font_size = self._config.font_size
        self._label.font_name = self._config.font_name
        self._label.color = self._config.color

    def on_config_changed(self, section: Optional[BaseModel]) -> None:
        self._update_config()

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> None:
        if self._deps is None:
            return

        self._x = x
        self._y = y
        w, h = self._deps.get_size()
        z = self._deps.viewport.screen_to_complex(x, y, w, h)

        self._label.text = f"z={z}"

    def draw(self) -> None:
        if self._deps is None:
            return

        x = self._x + self._config.x_pad
        y = self._y + self._config.y_pad

        self._label.x = int(x)
        self._label.y = int(y)
        self._label.draw()
