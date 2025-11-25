from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

from kit_mandelbrot.domain.viewport import Viewport
from kit_mandelbrot.services.cmd_engine import Command, CommandResult
from kit_mandelbrot.ui.theme import AppTheme


@dataclass
class UIContext:
    get_size: Callable[[], tuple[int, int]]
    execute_command: Callable[[str], CommandResult]
    get_command: Callable[[str], Optional[Command]]
    prompt_suggest: Callable[[str], list[str]]
    update_viewport: Callable[[Viewport], None]
    viewport: Viewport
    theme: AppTheme
