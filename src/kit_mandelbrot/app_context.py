from dataclasses import dataclass
from typing import Callable

from kit_mandelbrot.domain.viewport import Viewport

import moderngl

from kit_mandelbrot.rendering.pipeline import RenderPipeline
from kit_mandelbrot.services.fractal_engine import FractalEngine
from kit_mandelbrot.rendering.texture_presenter import TexturePresenter


@dataclass
class AppContext:
    gl_ctx: moderngl.Context
    presenter: TexturePresenter
    pipeline: RenderPipeline
    engine: FractalEngine
    update_viewport: Callable[[Viewport], None]
    quit: Callable[[], None]
