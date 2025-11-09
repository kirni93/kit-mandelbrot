from __future__ import annotations
import moderngl
from kit_mandelbrot.rendering.quad import FullscreenQuad
from kit_mandelbrot.rendering.texture_presenter import TexturePresenter
from typing import Any, cast


def set_sampler_unit(prog: moderngl.Program, name: str, unit: int) -> None:
    if name in prog:
        member = cast(Any, prog[name])  # moderngl member has .value at runtime
        member.value = unit


class RenderPipeline:
    def __init__(
        self,
        ctx: moderngl.Context,
        present_prog: moderngl.Program,
        quad: FullscreenQuad,
        presenter: TexturePresenter,
    ) -> None:
        self.ctx = ctx
        self.program = present_prog
        self.quad = quad
        self.presenter = presenter

    def draw(self) -> None:
        self.presenter.use(0)
        set_sampler_unit(self.program, "tex", 0)
        self.quad.draw()
