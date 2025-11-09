from __future__ import annotations
import moderngl
from typing import Tuple
import numpy as np


class TexturePresenter:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.texture: moderngl.Texture | None = None

    def ensure_size(self, size: Tuple[int, int]) -> None:
        w, h = size
        if self.texture is None or self.texture.size != (w, h):
            self.texture = self.ctx.texture(
                (w, h), components=1, dtype="f4"
            )  # RGB for gradient
            self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)

    def upload(self, image_f32: np.ndarray) -> None:
        """image_f32: shape (H, W), dtype float32, values 0..1"""
        assert image_f32.dtype == np.float32
        h, w = image_f32.shape
        self.ensure_size((w, h))

        assert self.texture is not None
        # GL expects row 0 = bottom; flip if your engine returns top-first
        self.texture.write(np.flipud(image_f32).tobytes())

    def use(self, unit: int = 0) -> None:
        if self.texture is not None:
            self.texture.use(unit)
