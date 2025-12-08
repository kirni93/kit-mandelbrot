from __future__ import annotations
import moderngl
from kit_mandelbrot.rendering.quad import FullscreenQuad
from kit_mandelbrot.rendering.texture_presenter import TexturePresenter
from typing import Any, cast

from importlib.resources import files


def set_sampler_unit(prog: moderngl.Program, name: str, unit: int) -> None:
    if name in prog:
        member = cast(Any, prog[name])  # moderngl member has .value at runtime
        member.value = unit


class RenderPipeline:
    def __init__(
        self,
        ctx: moderngl.Context,
        presenter: TexturePresenter,
        *,
        max_iter: int = 100,
        smooth: bool = True,
    ) -> None:
        self.ctx = ctx
        self.presenter = presenter

        # Load and compile presentation shaders here
        vs_src = (files("kit_mandelbrot.shaders") / "present.vert.glsl").read_text(
            "utf-8"
        )
        fs_src = (
            files("kit_mandelbrot.shaders") / "present_color.frag.glsl"
        ).read_text("utf-8")
        self.program = ctx.program(vertex_shader=vs_src, fragment_shader=fs_src)

        # Fullscreen quad uses this program
        self.quad = FullscreenQuad(ctx, self.program)

        # Cache uniforms (Pyright-friendly)
        self._u_tex = cast(moderngl.Uniform, self.program["tex"])
        self._u_use_smooth = cast(moderngl.Uniform, self.program["use_smooth"])
        self._u_max_iter = cast(moderngl.Uniform, self.program["max_iter"])

        # Initial values
        self.set_max_iter(max_iter)
        self.set_smooth(smooth)

    # Public API for the rest of the app

    def set_smooth(self, enabled: bool) -> None:
        self._u_use_smooth.value = int(enabled)

    def set_max_iter(self, max_iter: int) -> None:
        self._u_max_iter.value = int(max_iter)

    def draw(self) -> None:
        # Bind texture to unit 0 and tell shader
        self.presenter.use(0)
        self._u_tex.value = 0
        self.quad.draw()
