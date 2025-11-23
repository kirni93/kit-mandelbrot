from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from kit_mandelbrot.domain.viewport import Viewport
from kit_mandelbrot.services.cmd_engine import CommandResult
from kit_mandelbrot.ui.theme import AppTheme


@dataclass
class UIContext:
    get_size: Callable[[], tuple[int, int]]
    update_viewport: Callable[[Viewport], None]
    execute_command: Callable[[str], CommandResult]
    viewport: Viewport
    theme: AppTheme
