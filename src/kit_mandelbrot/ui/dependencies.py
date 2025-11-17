from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from kit_mandelbrot.domain.viewport import Viewport
from kit_mandelbrot.ui.theme import AppTheme


@dataclass(frozen=True)
class UIDeps:
    get_size: Callable[[], tuple[int, int]]
    viewport: Viewport
    theme: AppTheme
