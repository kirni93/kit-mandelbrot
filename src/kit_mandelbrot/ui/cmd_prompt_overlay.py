from __future__ import annotations

from typing import Optional

from pyglet import clock
from pyglet.text import Label
from pyglet.shapes import BorderedRectangle
from pyglet.window import key

from kit_mandelbrot.app_context import AppContext

from .ui_context import UIContext
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel


class CmdPromptOverlayConfig(BaseModel):
    pad: float = 0.4
    y_pos: float = 0.5
    width: float = 0.5
    prompt_symbol: str = ":>"
    caret_symbols: str = "| "
    caret_blink_interval_s: float = 0.5


class CmdPromptOverlay(UIElement):
    def __init__(self, config: CmdPromptOverlayConfig) -> None:
        super().__init__()

        self._config = config
        self._app_ctx: Optional[AppContext] = None
        self._deps: Optional[UIContext] = None
        self._window: Optional[Window] = None

        self._active = False

        self._bg = BorderedRectangle(0, 0, 0, 0, border=1)
        self._prompt = Label("", 0, 0)
        self._caret_idx = 0

        self._buffer = "Some test text for testing the text. It's for testing the text."

    def _update_config(self) -> None:
        pass

    def _build_components(self) -> None:
        if self._deps is None or self._window is None:
            return

        theme = self._deps.theme

        width, height = self._deps.get_size()

        # --- sizes ---
        font_size = theme.font_main_size
        pad = int(font_size * self._config.pad)

        w = int(width * self._config.width)
        h = int(pad * 2 + font_size)  # bg height in px

        # --- center position in screen coords ---
        x_center = width / 2
        y_center = height * self._config.y_pos  # normalized [0..1]

        # convert centers to bottom-left corner for Rectangle
        x = int(x_center - w / 2)
        y = int(y_center - h / 2)

        # --- background rectangle ---
        self._bg.color = theme.panel_bg
        self._bg.border_color = theme.panel_border
        self._bg.border = max(1, font_size // 10)
        self._bg.width = w
        self._bg.height = h
        self._bg.x = x
        self._bg.y = y

        # --- label (prompt) ---
        self._prompt.font_name = theme.mono_font or theme.font
        self._prompt.font_size = font_size

        # leave some padding on both sides for the text
        self._prompt.width = self._bg.width - 2 * pad

        # anchor left/center so y is the vertical center of the text
        self._prompt.anchor_x = "left"
        self._prompt.anchor_y = "center"

        self._prompt.x = self._bg.x + pad
        self._prompt.y = self._bg.y + self._bg.height / 2

        self._prompt.color = theme.text_primary

    def mount(self, window: Window, ctx: UIContext) -> None:
        self._deps = ctx
        self._window = window

        self._update_config()

    def unmount(self, window: Window) -> None:
        self._deps = None
        self._window = None

    def _activate(self) -> None:
        self._build_components()
        clock.schedule_interval(self._blink_caret, self._config.caret_blink_interval_s)
        self._active = True

    def _deactivate(self) -> None:
        clock.unschedule(self._blink_caret)
        self._active = False

    def _execute_cmd(self) -> None:
        if self._deps is None:
            return

        print(self._buffer)

        result = self._deps.execute_command(self._buffer)

        self._buffer = ""

        print(result.message)

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

    def on_resize(self, width: int, height: int) -> None:
        self._build_components()

    def on_text(self, text: str) -> None:
        if not self._active:
            return

        self._buffer += text

    def on_config_changed(self, section: Optional[BaseModel]) -> None:
        self._update_config()

    def on_theme_changed(self) -> None:
        self._build_components()

    def _blink_caret(self, dt: float) -> None:
        self._caret_idx = (self._caret_idx + 1) % len(self._config.caret_symbols)

    def draw(self) -> None:
        if self._deps is None:
            return

        if self._active:
            caret = self._config.caret_symbols[self._caret_idx]

            self._prompt.text = f"{self._config.prompt_symbol}{self._buffer}{caret}"
            self._bg.draw()
            self._prompt.draw()
