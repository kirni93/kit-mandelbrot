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


class ViewportOverlay(UIElement):
    def __init__(self, config: ViewportOverlayConfig) -> None:
        super().__init__()

        self._config = config
        self._deps = None

        self._headline = pyglet.text.Label(
            text="Viewport:", x=self._config.x_pad, y=self._config.y_pad
        )

        self._update_config()
        self._update_theme()

    def _update_config(self) -> None:
        pass

    def _update_theme(self) -> None:
        if self._deps is None:
            return

        theme = self._deps.theme

        self._headline.font_name = theme.font
        self._headline.font_size = theme.font_heading_size
        self._headline.color = theme.text_primary

    def mount(self, window: Window, deps: UIDeps) -> None:
        self._deps = deps

    def unmount(self, window: Window) -> None:
        self._deps = None

    def on_config_changed(self, section: Optional[BaseModel]) -> None:
        self._update_config()

    def on_theme_changed(self) -> None:
        self._update_theme()

    def draw(self) -> None:
        self._headline.draw()
