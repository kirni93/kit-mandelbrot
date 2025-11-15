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
            # Single-channel float texture (R32F), used as a stability / intensity map
            self.texture = self.ctx.texture((w, h), components=4, dtype="f4")
            self.texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.texture.repeat_x = False
            self.texture.repeat_y = False

    def upload(self, image_f32: np.ndarray) -> None:
        """image_f32: shape (H, W), dtype float32, values 0..1"""
        assert image_f32.dtype == np.float32
        h, w = image_f32.shape
        self.ensure_size((w, h))
        assert self.texture is not None

        # Expand grayscale → RGBA so old CPU path still works
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[..., 0] = image_f32  # R = stability
        rgba[..., 3] = 1.0  # A = 1

        self.texture.write(np.flipud(rgba).tobytes())

    def use(self, unit: int = 0) -> None:
        if self.texture is not None:
            self.texture.use(unit)
