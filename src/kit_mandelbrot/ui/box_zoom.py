from __future__ import annotations
from typing import Optional

import math
from pyglet.shapes import BorderedRectangle, Circle, Line
from pyglet.text import Label


from .ui_context import UIContext
from .types import UIElement
from pyglet.window import Window
from pydantic import BaseModel, field_validator
from kit_mandelbrot.domain.viewport import (
    viewport_from_points,
    screen_to_complex,
)
from pyglet.window import mouse


def fmt_val(v: float, decimals: int = 5) -> str:
    """Format a single float, switching to scientific notation when appropriate."""
    abs_v = abs(v)
    if abs_v != 0 and (abs_v < 1e-4 or abs_v >= 1e4):
        # Scientific notation
        return f"{v:.{decimals}e}"
    else:
        # Normal decimal
        return f"{v:.{decimals}f}"


def format_complex(z: complex, re_dec: int = 3, im_dec: int = 3) -> str:
    """Human-friendly complex number formatting used in modern tools."""
    re = fmt_val(z.real, re_dec)
    im = fmt_val(z.imag, im_dec)
    sign = "+" if z.imag >= 0 else "-"
    return f"{re} {sign} {im.lstrip('-')}i"


def decimals_for_axis(
    span: float, base_decimals: int = 3, min_decimals: int = 0, max_decimals: int = 10
) -> int:
    span = abs(span)

    if span <= 0:
        return base_decimals

    order = math.floor(math.log10(span))

    decimals = base_decimals - order

    return max(min_decimals, min(max_decimals, decimals))


class Callout:
    def __init__(self, ctx: UIContext) -> None:
        self._ctx = ctx

        self.text = ""
        self.anchor_x = 0
        self.anchor_y = 0
        self.anchor_radius = 3

        self.font = ctx.theme.font
        self.font_size = ctx.theme.font_main_size
        self.text_color = ctx.theme.text_primary
        self.bg_color = ctx.theme.panel_bg
        self.border_color = ctx.theme.panel_border_active

        self.padding = 4
        self.margin = 6
        self.offset_x = 20
        self.offset_y = 20

        self.opacity = 200

        self._anchor = Circle(0, 0, self.anchor_radius)
        self._pointer = Line(0, 0, 0, 0)
        self._background = BorderedRectangle(0, 0, 0, 0, border=1)
        self._label = Label("", x=0, y=0, anchor_x="left", anchor_y="bottom")

    def draw(self) -> None:
        if not self.text:
            return

        win_w, win_h = self._ctx.get_size()

        self._label.text = self.text
        self._label.font_name = self.font
        self._label.font_size = self.font_size
        self._label.color = self.text_color

        text_w = self._label.content_width
        text_h = self._label.content_height

        callout_w = text_w + 2 * self.padding
        callout_h = text_h + 2 * self.padding

        # default callout position is simply adding the offset
        cx = self.anchor_x + self.offset_x
        cy = self.anchor_y + self.offset_y

        # flip horizontally of outside of window
        if cx + callout_w > win_w - self.margin:
            cx = self.anchor_x - self.offset_x - callout_w

        # flip vertically if out of bounds
        if cy + callout_h > win_h - self.margin:
            cy = self.anchor_y - self.offset_y - callout_h

        # anchor directly to the coordinates
        self._anchor.x = self.anchor_x
        self._anchor.y = self.anchor_y
        self._anchor.radius = self.anchor_radius
        self._anchor.color = self.border_color
        self._anchor.opacity = self.opacity

        self._background.x = cx
        self._background.y = cy
        self._background.width = callout_w
        self._background.height = callout_h
        self._background.color = self.bg_color
        self._background.border_color = self.border_color
        self._background.opacity = self.opacity

        self._label.x = cx + self.padding
        self._label.y = cy + self.padding

        ax, ay = self.anchor_x, self.anchor_y

        bx = min(max(ax, cx), cx + callout_w)
        by = min(max(ay, cy), cy + callout_h)

        self._pointer.x = ax
        self._pointer.y = ay
        self._pointer.x2 = bx
        self._pointer.y2 = by
        self._pointer.color = self.border_color
        self._pointer.opacity = self.opacity

        self._background.draw()
        self._pointer.draw()
        self._anchor.draw()
        self._label.draw()


class BoxZoomConfig(BaseModel):
    opacity: int = 200
    point_radius: int = 5
    min_box_width: int = 30
    min_box_height: int = 30

    @field_validator("opacity")
    @classmethod
    def check_opacity(cls, v: int) -> int:
        if not 0 <= v <= 255:
            raise ValueError("opacity must be between 0 und 255")
        return v


class BoxZoom(UIElement):
    def __init__(self, config: BoxZoomConfig) -> None:
        self._ctx: Optional[UIContext] = None
        self._x: int = 0
        self._y: int = 0
        self._x_start: int = 0
        self._y_start: int = 0
        self._x_end: int = 0
        self._y_end: int = 0
        self._dragging: bool = False
        self._config = config

        self._box = BorderedRectangle(0, 0, 0, 0, border=1)
        self._start_callout: Optional[Callout]
        self._end_callout: Optional[Callout]

    def mount(self, window: Window, ctx: UIContext) -> None:
        self._ctx = ctx
        self._start_callout = Callout(ctx)
        self._end_callout = Callout(ctx)

        self._update_config()

    def unmount(self, window: Window) -> None:
        self._ctx = None
        self._start_callout = None
        self._end_callout = None

    def _update_config(self) -> None:
        if self._ctx is None:
            return

        self._build_components()

    def on_config_changed(self, section: Optional[BaseModel]) -> None:
        self._update_config()

    def _build_components(self) -> None:
        if self._ctx is None:
            return

        self._box.color = self._ctx.theme.panel_bg
        self._box.border_color = self._ctx.theme.panel_border_active
        self._box.opacity = self._config.opacity

    def on_mouse_drag(
        self, x: int, y: int, dx: int, dy: int, buttons, modifiers
    ) -> None:
        if not (buttons & mouse.LEFT):
            return

        if not self._dragging:
            self._x_start = x
            self._y_start = y

            self._dragging = True

        self._x = x
        self._y = y

    def _set_viewport(self) -> None:
        if self._ctx is None:
            return

        screen_w, screen_h = self._ctx.get_size()

        min_x = min(self._x_start, self._x_end)
        max_x = max(self._x_start, self._x_end)
        min_y = min(self._y_start, self._y_end)
        max_y = max(self._y_start, self._y_end)

        if abs(max_x - min_x) < self._config.min_box_width:
            return

        if abs(max_y - min_y) < self._config.min_box_height:
            return

        c1 = screen_to_complex(self._ctx.viewport, min_x, min_y, screen_w, screen_h)
        c2 = screen_to_complex(self._ctx.viewport, max_x, max_y, screen_w, screen_h)

        vp = viewport_from_points(c1, c2)

        self._ctx.update_viewport(vp)

    def on_mouse_release(self, x: int, y: int, button, modifiers) -> None:
        if not (button & mouse.LEFT):
            return

        if not self._dragging:
            return

        self._dragging = False
        self._x_end = x
        self._y_end = y

        self._set_viewport()

    def draw(self) -> None:
        if self._ctx is None:
            return

        if self._end_callout is None:
            return

        if self._start_callout is None:
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

            screen_w, screen_h = self._ctx.get_size()

            re_span = self._ctx.viewport.re_max - self._ctx.viewport.re_min
            re_decimals = decimals_for_axis(re_span)

            im_span = self._ctx.viewport.imag_max - self._ctx.viewport.imag_min
            im_decimals = decimals_for_axis(im_span)

            c_start = screen_to_complex(
                self._ctx.viewport, self._x_start, self._y_start, screen_w, screen_h
            )

            start_text = format_complex(
                z=c_start, re_dec=re_decimals, im_dec=im_decimals
            )

            self._start_callout.anchor_x = self._x_start
            self._start_callout.anchor_y = self._y_start
            self._start_callout.text = start_text

            self._start_callout.draw()

            c = screen_to_complex(
                self._ctx.viewport, self._x, self._y, screen_w, screen_h
            )

            end_text = format_complex(z=c, re_dec=re_decimals, im_dec=im_decimals)

            self._end_callout.anchor_x = self._x
            self._end_callout.anchor_y = self._y
            self._end_callout.text = end_text

            self._end_callout.draw()
