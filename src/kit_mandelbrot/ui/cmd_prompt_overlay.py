from __future__ import annotations

import re
from typing import Optional

from pyglet.text import Label
from pyglet.shapes import Rectangle
from pyglet.window import key
from .dependencies import UIDeps
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel


class CmdPromptOverlayConfig(BaseModel):
    x_pad: int = 12
    y_pad: int = 12
    bg_pad: int = 4


class CmdPromptOverlay(UIElement):
    def __init__(self, config: CmdPromptOverlayConfig) -> None:
        super().__init__()

        self._config = config
        self._deps: Optional[UIDeps] = None
        self._window: Optional[Window] = None

        self._active = False

        self._bg = Rectangle(0, 0, 0, 0)
        self._prompt = Label("", 0, 0)

        self._buffer = "Some test text for testing the text. It's for testing the text."

    def _update_config(self) -> None:
        pass

    def _build_components(self) -> None:
        if self._deps is None or self._window is None:
            return

        theme = self._deps.theme

        w = self._window.width

        self._bg.color = theme.panel_bg
        self._bg.height = self._config.bg_pad * 2 + theme.font_main_size
        self._bg.width = w
        self._bg.x = 0
        self._bg.y = 0

        self._prompt.font_name = theme.font
        self._prompt.font_size = theme.font_main_size
        self._prompt.width = w - 2 * self._config.bg_pad
        self._prompt.y = self._config.bg_pad
        self._prompt.x = self._config.bg_pad
        self._prompt.color = theme.text_primary

    def mount(self, window: Window, deps: UIDeps) -> None:
        self._deps = deps
        self._window = window

        self._update_config()
        self._build_components()

    def unmount(self, window: Window) -> None:
        self._deps = None
        self._window = None

    def _activate(self) -> None:
        self._active = True

    def _deactivate(self) -> None:
        self._active = False

    def _execute_cmd(self) -> None:
        print(self._buffer)

        cmd = self._buffer.strip()

        parts = cmd.split()

        command = parts[0]

        if command in ("vp", "viewport"):
            self._handle_viewport_command(parts[1:])

        self._buffer = ""

    def _handle_viewport_command(self, args: list[str]) -> None:
        if self._deps is None:
            return

        print(f"try vp with params: {args}")

        if len(args) != 4:
            print("params not matching")
            return

        try:
            re_min, re_max, im_min, im_max = [float(a) for a in args]
        except ValueError:
            print("Could not cast")
            return

        if re_min > re_max:
            print("real min bigger than max")
            return

        if im_min > im_max:
            print("real min bigger than max")
            return

        vp = self._deps.viewport

        vp.re_min = re_min
        vp.re_max = re_max
        vp.imag_min = im_min
        vp.imag_max = im_max

        self._deps.update_viewport()

    def on_key_press(self, symbol: int, modifiers: int) -> None:
        if not self._active:
            if symbol == key.COLON:
                self._activate()
                return

        if symbol == key.ESCAPE:
            self._deactivate()
            return

        if symbol == key.ENTER:
            self._execute_cmd()
            self._deactivate()
            return

        if symbol == key.BACKSPACE:
            self._buffer = self._buffer[:-1]
            return

    def on_text(self, text: str) -> None:
        if not self._active:
            return

        self._buffer += text

    def on_config_changed(self, section: Optional[BaseModel]) -> None:
        self._update_config()

    def on_theme_changed(self) -> None:
        self._build_components()

    def _update_layout(self) -> None:
        if self._window is None or self._deps is None:
            return

        self._bg.width = self._window.width

    def draw(self) -> None:
        if self._deps is None:
            return

        if self._active:
            self._update_layout()

            self._prompt.text = self._buffer

            self._bg.draw()
            self._prompt.draw()
