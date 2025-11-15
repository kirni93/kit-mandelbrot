from __future__ import annotations

import pyglet
from typing import Optional
from .dependencies import UIDeps
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel


class ViewportOverlayConfig(BaseModel):
    x_pad: int = 12
    y_pad: int = 12
    font_size: int = 12
    font_name: str = "Menlo"
    color: tuple[int, int, int, int] = (230, 230, 230, 255)


class ViewportOverlay(UIElement):
    def __init__(self, config: ViewportOverlayConfig) -> None:
        super().__init__()

        self._config = config

        self._headline = pyglet.text.Label(
            text="Viewport:", x=self._config.x_pad, y=self._config.y_pad
        )

        self._update_config()

    def _update_config(self) -> None:
        self._headline.color = self._config.color
        self._headline.font_size = self._config.font_size
        self._headline.font_name = self._config.font_name

    def mount(self, window: Window, deps: UIDeps) -> None:
        self._deps = deps

    def unmount(self, window: Window) -> None:
        self._deps = None

    def on_config_changed(self, section: Optional[BaseModel]) -> None:
        self._update_config()

    def draw(self) -> None:
        self._headline.draw()
