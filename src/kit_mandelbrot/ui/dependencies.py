from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from kit_mandelbrot.domain import viewport


@dataclass(frozen=True)
class UIDeps:
    get_size: Callable[[], tuple[int, int]]
    viewport: viewport.Viewport
