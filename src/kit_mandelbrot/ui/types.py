from typing import Protocol, runtime_checkable, Optional
from pydantic import BaseModel
from pyglet.window import Window

from kit_mandelbrot.ui.ui_context import UIContext


@runtime_checkable
class UIElement(Protocol):
    def mount(self, window: Window, ctx: UIContext) -> None: ...

    def unmount(self, window: Window) -> None: ...

    def on_config_changed(self, section: Optional[BaseModel]) -> None: ...

    def draw(self) -> None: ...
