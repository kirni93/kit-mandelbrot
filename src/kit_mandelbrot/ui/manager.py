from __future__ import annotations
from typing import List
from pyglet.window import Window

from kit_mandelbrot.ui.ui_context import UIContext
from .types import UIElement


class UIManager:
    def __init__(self, window: Window, ctx: UIContext) -> None:
        self.window = window
        self.ctx = ctx
        self._elements: List[UIElement] = []

    def add(self, element: UIElement) -> None:
        element.mount(self.window, self.ctx)
        self.window.push_handlers(element)
        self._elements.append(element)

    def config_changed(self) -> None:
        for e in self._elements:
            e.on_config_changed(section=None)

    def draw(self) -> None:
        for e in self._elements:
            e.draw()
