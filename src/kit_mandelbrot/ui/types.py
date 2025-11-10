from typing import Protocol, runtime_checkable, Optional
from pyglet.window import Window
from .dependencies import UIDeps


@runtime_checkable
class UIElement(Protocol):
    def mount(self, window: Window, deps: UIDeps) -> None: ...

    def unmount(self, window: Window) -> None: ...

    def draw(self) -> None: ...
