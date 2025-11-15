from __future__ import annotations
import moderngl
import numpy as np


class FullscreenQuad:
    def __init__(self, ctx: moderngl.Context, prog: moderngl.Program) -> None:
        verts = np.array(
            [
                -1,
                -1,
                0,
                0,
                1,
                -1,
                1,
                0,
                1,
                1,
                1,
                1,
                -1,
                1,
                0,
                1,
            ],
            dtype="f4",
        )
        idx = np.array([0, 1, 2, 2, 3, 0], dtype="i4")
        self._vbo = ctx.buffer(verts.tobytes())
        self._ibo = ctx.buffer(idx.tobytes())
        self._vao = ctx.vertex_array(
            prog, [(self._vbo, "2f 2f", "in_pos", "in_uv")], self._ibo
        )

    def draw(self) -> None:
        self._vao.render()
