from dataclasses import dataclass


@dataclass
class Viewport:
    re_min: float
    re_max: float
    imag_min: float
    imag_max: float

    def screen_to_complex(self, x: int, y: int, width: int, height: int) -> complex:
        """Convert screen coordinates to a complex-plane coordinate"""
        re = self.re_min + (x / width) * (self.re_max - self.re_min)
        imag = self.imag_max - (y / height) * (self.imag_max - self.imag_min)

        return complex(real=re, imag=imag)

    # --- zoom ---------------------------------------------------------------
    def zoom_viewport_at(self, x_frac: float, y_frac: float, zoom: float) -> None:
        """
        Zoom viewport around a fractional screen position (x_frac, y_frac).
        zoom < 1 => zoom in, zoom > 1 => zoom out. Modifies self in-place.
        Keeps the mouse position fixed in view.
        """
        re_span = self.re_max - self.re_min
        im_span = self.imag_max - self.imag_min

        # current complex coordinates under cursor
        re_cursor = self.re_min + x_frac * re_span
        im_cursor = self.imag_min + y_frac * im_span

        new_re_span = re_span * zoom
        new_im_span = im_span * zoom

        # place cursor at same fractional position after zoom
        self.re_min = re_cursor - x_frac * new_re_span
        self.re_max = self.re_min + new_re_span
        self.imag_min = im_cursor - y_frac * new_im_span
        self.imag_max = self.imag_min + new_im_span

    # --- zoom into a box selected on screen ---------------------------------
    def zoom_box(self, xf0: float, yf0: float, xf1: float, yf1: float) -> None:
        """
        Zoom viewport to a box defined by two fractional corners (xf0,yf0) and (xf1,yf1).
        Fractions are 0..1 in both axes, bottom-left origin.
        """
        re_min, re_max = self.re_min, self.re_max
        im_min, im_max = self.imag_min, self.imag_max
        re_span = re_max - re_min
        im_span = im_max - im_min

        # Compute new boundaries in fractal coordinates
        self.re_min = min(xf0, xf1) * re_span + re_min
        self.re_max = max(xf0, xf1) * re_span + re_min
        self.imag_min = min(yf0, yf1) * im_span + im_min
        self.imag_max = max(yf0, yf1) * im_span + im_min

    # --- panning ------------------------------------------------------------
    def pan_by_frac(self, dx_frac: float, dy_frac: float) -> None:
        """
        Translate the viewport by dx_frac, dy_frac of its span.
        Positive dx moves right, positive dy moves up (screen coords: bottom-left).
        """
        re_span = self.re_max - self.re_min
        im_span = self.imag_max - self.imag_min
        self.re_min += dx_frac * re_span
        self.re_max += dx_frac * re_span
        self.imag_min += dy_frac * im_span
        self.imag_max += dy_frac * im_span

    # --- aspect utilities ----------------------------------------------------
    def get_spans(self):
        """Return (re_span, im_span)."""
        return (self.re_max - self.re_min, self.imag_max - self.imag_min)

    def set_imag_span_from_aspect(self, aspect: float) -> None:
        """
        Adjust imag_min/max to match a given aspect ratio (width/height).
        Useful when resetting.
        """
        re_span = self.re_max - self.re_min
        im_span = re_span / aspect
        im_center = (self.imag_min + self.imag_max) / 2.0
        self.imag_min = im_center - im_span / 2.0
        self.imag_max = im_center + im_span / 2.0
