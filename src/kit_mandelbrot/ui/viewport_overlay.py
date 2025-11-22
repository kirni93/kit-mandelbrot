from __future__ import annotations

import pyglet
from typing import Optional

from pyglet.text import Label
from .dependencies import UIDeps
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel


class ViewportOverlayConfig(BaseModel):
    x_pad: int = 12
    y_pad: int = 12
    line_spacing: int = 4


class ViewportOverlay(UIElement):
    def __init__(self, config: ViewportOverlayConfig) -> None:
        super().__init__()

        self._config = config
        self._deps: Optional[UIDeps] = None
        self._window: Optional[Window] = None

        self._re_line = Label()
        self._im_line = Label()

    def _update_config(self) -> None:
        pass

    def _build_components(self) -> None:
        if self._deps is None:
            return

        theme = self._deps.theme

        # Lines below heading
        self._re_line = pyglet.text.Label(
            text="",  # will be filled in draw()
            font_name=theme.font,
            font_size=theme.font_main_size,
            color=theme.text_muted,
            anchor_x="right",
            anchor_y="top",
        )

        self._im_line = pyglet.text.Label(
            text="",
            font_name=theme.font,
            font_size=theme.font_main_size,
            color=theme.text_muted,
            anchor_x="right",
            anchor_y="top",
        )

    def mount(self, window: Window, deps: UIDeps) -> None:
        self._deps = deps
        self._window = window

        self._update_config()
        self._build_components()

    def unmount(self, window: Window) -> None:
        self._deps = None
        self._window = None

    def on_config_changed(self, section: Optional[BaseModel]) -> None:
        self._update_config()

    def on_theme_changed(self) -> None:
        self._build_components()

    def _update_layout(self) -> None:
        """Reposition labels in top-right based on current window size."""
        if self._window is None or self._deps is None:
            return

        theme = self._deps.theme
        w, h = self._window.width, self._window.height

        x = w - self._config.x_pad
        y_top = h - self._config.y_pad

        y = y_top - theme.font_heading_size - self._config.line_spacing
        self._re_line.x = x
        self._re_line.y = y

        y -= theme.font_main_size + self._config.line_spacing
        self._im_line.x = x
        self._im_line.y = y

    def draw(self) -> None:
        if self._deps is None:
            return

        # Update text from current viewport
        vp = self._deps.viewport

        self._re_line.text = f"Re: [{vp.re_min:.6f}, {vp.re_max:.6f}]"
        self._im_line.text = f"Im: [{vp.imag_min:.6f}, {vp.imag_max:.6f}]"

        # Keep anchored to top-right even if window resized
        self._update_layout()

        # Draw labels
        self._re_line.draw()
        self._im_line.draw()
