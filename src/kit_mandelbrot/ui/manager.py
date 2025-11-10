from __future__ import annotations
from typing import List
from pyglet.window import Window
from .types import UIElement
from .dependencies import UIDeps


class UIManager:
    def __init__(self, window: Window, deps: UIDeps) -> None:
        self.window = window
        self.deps = deps
        self._elements: List[UIElement] = []

    def add(self, element: UIElement) -> None:
        element.mount(self.window, self.deps)
        self.window.push_handlers(element)
        self._elements.append(element)

    def draw(self) -> None:
        for e in self._elements:
            e.draw()
