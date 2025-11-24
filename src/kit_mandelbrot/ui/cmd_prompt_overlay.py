from __future__ import annotations

from typing import Optional

from pyglet import clock
from pyglet.text import Label
from pyglet.shapes import BorderedRectangle
from pyglet.window import key

from .ui_context import UIContext
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel


class CmdPromptOverlayConfig(BaseModel):
    pad: float = 0.4
    y_pos: float = 0.9
    width: float = 0.5
    prompt_symbol: str = ":>"
    caret_symbols: str = "| "
    caret_blink_interval_s: float = 0.5
    max_suggests: int = 10


class CmdPromptOverlay(UIElement):
    def __init__(self, config: CmdPromptOverlayConfig) -> None:
        super().__init__()

        self._config = config
        self._deps: Optional[UIContext] = None
        self._window: Optional[Window] = None

        self._active = False

        self._bg = BorderedRectangle(0, 0, 0, 0, border=1)
        self._prompt = Label("", 0, 0)
        self._caret_idx = 0

        self._buffer = "Some test text for testing the text. It's for testing the text."
        self._line_suggest = list()
        self._suggest_select: Optional[int] = None
        self._suggest_labels: list[Label] = []

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
        self._update_suggestions()
        self._build_components()
        clock.schedule_interval(self._blink_caret, self._config.caret_blink_interval_s)
        self._active = True

    def _deactivate(self) -> None:
        clock.unschedule(self._blink_caret)
        self._active = False

    def _build_suggestion_labels(self) -> None:
        if self._deps is None:
            return

        theme = self._deps.theme

        self._suggest_labels.clear()

        if not self._line_suggest:
            return

        font_size = theme.font_small_size
        font = theme.font
        pad = int(theme.font_main_size * self._config.pad)
        # start below the prompt background
        x = self._bg.x + pad
        base_y = self._bg.y - self._bg.height - pad // 2

        # how many suggestions to show at once
        max_items = self._config.max_suggests
        line_height = font_size + 2

        for i, text in enumerate(self._line_suggest[:max_items]):
            lbl = Label(
                text,
                x=x,
                y=base_y - i * line_height,
                font_name=font,
                font_size=font_size,
                anchor_x="left",
                anchor_y="bottom",
            )

            # selected suggestion gets accent color, others muted
            if self._suggest_select is not None and self._suggest_select == i:
                lbl.color = theme.text_accent
            else:
                lbl.color = theme.text_muted

            self._suggest_labels.append(lbl)

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

        is_shift = modifiers & key.MOD_SHIFT

        if symbol == key.ESCAPE:
            if self._suggest_select is None:
                self._deactivate()
            else:
                self._suggest_select = None
            return

        if symbol == key.BACKSPACE:
            self._buffer = self._buffer[:-1]
            return

        if symbol == key.ENTER:
            self._enter()
            return

        if symbol == key.TAB and not is_shift:
            self._tab_forward()
            print(self._suggest_select)
            return

        if symbol == key.TAB and is_shift:
            self._tab_backward()
            print(self._suggest_select)
            return

    def _enter(self) -> None:
        if self._suggest_select is None:
            self._execute_cmd()
            self._deactivate()
            return

        self._apply_suggest(self._suggest_select)

    def _tab_forward(self) -> None:
        if self._deps is None:
            return

        if not self._line_suggest:
            self._update_suggestions()

        if not self._line_suggest:
            return

        if len(self._line_suggest) == 1:
            self._apply_suggest(0)
            return

        if self._suggest_select is None:
            self._suggest_select = 0
        else:
            self._suggest_select = (self._suggest_select + 1) % len(self._line_suggest)

        self._build_suggestion_labels()

    def _tab_backward(self):
        if self._deps is None:
            return

        if not self._line_suggest:
            self._update_suggestions()

        if not self._line_suggest:
            return

        if self._suggest_select is None:
            self._suggest_select = len(self._line_suggest) - 1
        else:
            self._suggest_select = (self._suggest_select - 1) % len(self._line_suggest)

        self._build_suggestion_labels()

    def _apply_suggest(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._line_suggest):
            return

        suggestion = self._line_suggest[idx]

        parts = self._buffer.split() or [""]

        if self._buffer.endswith(" "):
            parts.append(suggestion)
        else:
            parts[-1] = suggestion

        self._buffer = " ".join(parts) + " "
        self._update_suggestions()

        self._suggest_select = None

    def on_resize(self, width: int, height: int) -> None:
        self._build_components()

    def on_text(self, text: str) -> None:
        if not self._active:
            return

        if text in ("\t", "\r", "\n"):
            return

        self._buffer += text

        self._suggest_select = None
        self._update_suggestions()

    def on_config_changed(self, section: Optional[BaseModel]) -> None:
        self._update_config()

    def on_theme_changed(self) -> None:
        self._build_components()

    def _blink_caret(self, dt: float) -> None:
        self._caret_idx = (self._caret_idx + 1) % len(self._config.caret_symbols)

    def _update_suggestions(self) -> None:
        if self._deps is None:
            return

        self._line_suggest = self._deps.prompt_suggest(self._buffer)
        self._build_suggestion_labels()
        print(self._line_suggest)

    def draw(self) -> None:
        if self._deps is None:
            return

        if self._active:
            caret = self._config.caret_symbols[self._caret_idx]

            self._prompt.text = f"{self._config.prompt_symbol}{self._buffer}{caret}"
            self._bg.draw()

            for label in self._suggest_labels:
                label.draw()

            self._prompt.draw()
