from __future__ import annotations

from typing import Optional

from pyglet import clock
from pyglet.text import Label
from pyglet.shapes import BorderedRectangle, Rectangle
from pyglet.window import key

from .ui_context import UIContext
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel

BACKSPACE_REPEAT_DELAY = 0.4  # seconds before repeat starts
BACKSPACE_REPEAT_INTERVAL = 0.04  # seconds between repeats


class CmdPromptOverlayConfig(BaseModel):
    pad: float = 0.4
    y_pos: float = 0.9
    width: float = 0.5
    prompt_symbol: str = "> "
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
        self._title_label = Label("", 0, 0)
        self._title_bg = Rectangle(0, 0, 0, 0)
        self._prompt_symbol = Label("", 0, 0, 0, 0)
        self._prompt = Label("", 0, 0)
        self._caret_idx = 0

        self._buffer = ""
        self._line_suggest = list()
        self._suggest_select: Optional[int] = None
        self._suggest_labels: list[Label] = []
        self._suggest_bg = BorderedRectangle(0, 0, 0, 0, border=1)
        self._help_bg = BorderedRectangle(0, 0, 0, 0, border=1)
        self._help_label = Label("", 0, 0)

    def _update_config(self) -> None:
        pass

    def _position_help_label(self, font_size: int, pad_y: int) -> None:
        if not self._help_label.text:
            return

        pad_x = int(font_size * 0.7)

        self._help_label.x = self._help_bg.x + pad_x
        self._help_label.y = self._help_bg.y + self._help_bg.height - pad_y

    def _build_help(self) -> None:
        if self._deps is None:
            return

        theme = self._deps.theme
        font = theme.font
        font_size = theme.font_main_size

        # panel geometry (width, x; height will be adjusted later)
        self._help_bg.x = self._suggest_bg.x + self._suggest_bg.width + 1
        self._help_bg.y = self._suggest_bg.y
        self._help_bg.width = self._bg.width - self._suggest_bg.width - 1

        self._help_bg.color = theme.panel_bg
        self._help_bg.border_color = theme.panel_border
        self._help_bg.border = self._suggest_bg.border

        if self._suggest_select is None or not self._line_suggest:
            self._help_label.text = ""
            self._help_bg.height = self._suggest_bg.height  # or 0
            return

        name = self._line_suggest[self._suggest_select]
        cmd = self._deps.get_command(name)
        if cmd is None:
            self._help_label.text = ""
            self._help_bg.height = self._suggest_bg.height
            return

        usage = cmd.usage or cmd.name
        summary = cmd.summary or ""
        nl = "\n"
        help_text = f"usage: {usage}{nl}{nl}{summary}"

        pad_x = int(font_size * 0.7)
        pad_y = int(font_size * 0.4)

        self._help_label.text = help_text
        self._help_label.font_name = font
        self._help_label.font_size = font_size
        self._help_label.color = theme.text_primary

        self._help_label.anchor_x = "left"
        self._help_label.anchor_y = "top"

        self._help_label.width = int(self._help_bg.width - 2 * pad_x)
        self._help_label.multiline = True

        # provisional height based on text
        text_height = self._help_label.content_height
        self._help_bg.height = pad_y * 2 + text_height

    def _build_components(self) -> None:
        if self._deps is None or self._window is None:
            return

        theme = self._deps.theme

        width, height = self._deps.get_size()

        # --- sizes ---
        font_size = theme.font_main_size
        pad = int(font_size * self._config.pad)

        w = int(width * self._config.width)
        h = int(pad * 2 + font_size * 2)  # bg height in px

        # --- center position in screen coords ---
        x_center = width / 2
        y_center = height * self._config.y_pos  # normalized [0..1]

        # convert centers to bottom-left corner for Rectangle
        x = int(x_center - w / 2)
        y = int(y_center - h / 2)

        # --- background rectangle ---
        self._bg.color = theme.panel_bg
        self._bg.border_color = theme.panel_border_active
        self._bg.border = max(1, font_size // 10)
        self._bg.width = w
        self._bg.height = h
        self._bg.x = x
        self._bg.y = y

        # --- prompt
        self._prompt.font_name = theme.mono_font or theme.font
        self._prompt.font_size = font_size

        self._prompt_symbol.font_name = self._prompt.font_name
        self._prompt_symbol.font_size = self._prompt.font_size
        self._prompt_symbol.text = self._config.prompt_symbol
        self._prompt_symbol.color = self._bg.border_color

        self._prompt.anchor_x = "left"
        self._prompt.anchor_y = "center"

        self._prompt_symbol.anchor_x = self._prompt.anchor_x
        self._prompt_symbol.anchor_y = self._prompt.anchor_y

        self._prompt_symbol.x = self._bg.x + pad
        self._prompt.x = self._prompt_symbol.x + self._prompt_symbol.content_width
        self._prompt.y = self._bg.y + self._bg.height / 2
        self._prompt_symbol.y = self._prompt.y

        self._prompt.color = theme.text_primary

        # title label
        self._title_label.font_size = theme.font_small_size
        self._title_label.font_name = theme.font
        self._title_label.text = "Cmdline"
        self._title_label.anchor_x = "center"
        self._title_label.anchor_y = "center"
        self._title_label.color = self._bg.border_color
        # compute title paddings
        title_pad_x = int(self._title_label.font_size * 0.8)
        title_pad_y = int(self._title_label.font_size * 0.4)

        # label will sit exactly on the top border line
        border_y = self._bg.y + self._bg.height

        # center horizontally over the bg
        self._title_label.x = x_center
        self._title_label.y = border_y

        # get intrinsic label size
        title_w = self._title_label.content_width
        title_h = self._title_label.content_height

        # title background rectangle, centered on the border line
        bg_w = int(title_w + 2 * title_pad_x)
        bg_h = int(title_h + 2 * title_pad_y)

        self._title_bg.width = bg_w
        self._title_bg.height = bg_h
        self._title_bg.color = self._bg.color
        self._title_bg.x = int(x_center - bg_w / 2)
        self._title_bg.y = int(border_y - bg_h / 2)

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
            self._suggest_bg.height = 0
            self._help_bg.height = 0
            self._help_label.text = ""
            return

        mono_font = theme.font
        font_size = theme.font_main_size

        pad_x = int(font_size * 0.7)
        pad_y = int(font_size * 0.4)
        line_height = font_size + pad_y

        items = self._line_suggest[: self._config.max_suggests]
        count = len(items)

        # suggestion panel size & position
        self._suggest_bg.x = self._bg.x
        self._suggest_bg.width = self._bg.width / 3
        self._suggest_bg.height = pad_y * 2 + count * line_height
        self._suggest_bg.y = self._bg.y - self._suggest_bg.height - 1

        self._suggest_bg.color = theme.panel_bg
        self._suggest_bg.border_color = theme.panel_border
        self._suggest_bg.border = max(1, font_size // 10)

        base_x = self._suggest_bg.x + pad_x
        base_y = self._suggest_bg.y + self._suggest_bg.height - pad_y - font_size

        for i, item in enumerate(items):
            lbl = Label(
                item,
                x=base_x,
                y=base_y - i * line_height,
                font_name=mono_font,
                font_size=font_size,
                anchor_x="left",
                anchor_y="baseline",
            )

            lbl.color = (
                theme.text_accent if self._suggest_select == i else theme.text_primary
            )
            self._suggest_labels.append(lbl)

        # build help panel (computes geometry & text, but not final y of label)
        self._build_help()

        # unify heights
        max_height = max(self._help_bg.height, self._suggest_bg.height)
        self._suggest_bg.height = max_height
        self._help_bg.height = max_height
        self._suggest_bg.y = self._bg.y - max_height - 1
        self._help_bg.y = self._suggest_bg.y

        # now that heights are final, position the help label
        self._position_help_label(font_size, pad_y)

    def _execute_cmd(self) -> None:
        if self._deps is None:
            return

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
            return

        if symbol == key.TAB and is_shift:
            self._tab_backward()
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

    def draw(self) -> None:
        if self._deps is None:
            return

        if self._active:
            caret = self._config.caret_symbols[self._caret_idx]

            self._prompt.text = f"{self._buffer}{caret}"
            self._bg.draw()
            self._title_bg.draw()
            self._title_label.draw()

            if self._suggest_labels:
                self._suggest_bg.draw()
                self._help_bg.draw()
                self._help_label.draw()

            for label in self._suggest_labels:
                label.draw()

            self._prompt_symbol.draw()
            self._prompt.draw()
