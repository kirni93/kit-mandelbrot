from dataclasses import dataclass
from typing import NamedTuple

DEFAULT_FONT = "Menlo"
DEFAULT_MONO_FONT = "Fira Code"
DEFAULT_FONT_SMALL_SIZE = 16
DEFAULT_FONT_MAIN_SIZE = 20
DEFAULT_FONT_HEADING_SIZE = 28
DEFAULT_FONT_TITLE_SIZE = 36


class Color(NamedTuple):
    r: int
    g: int
    b: int
    a: int = 255

    @classmethod
    def from_hex(cls, hex_str: str, alpha: int = 255) -> "Color":
        hex_str = hex_str.lstrip("#")
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)

        return cls(r, g, b, alpha)

    def to_hex(self, include_alpha: bool = False) -> str:
        """Return #RRGGBB or #RRGGBBAA."""
        if include_alpha:
            return "#{:02x}{:02x}{:02x}{:02x}".format(self.r, self.g, self.b, self.a)
        return "#{:02x}{:02x}{:02x}".format(self.r, self.g, self.b)


@dataclass(frozen=True)
class Base16Palette:
    scheme: str
    author: str
    base00: Color
    base01: Color
    base02: Color
    base03: Color
    base04: Color
    base05: Color
    base06: Color
    base07: Color
    base08: Color
    base09: Color
    base0A: Color
    base0B: Color
    base0C: Color
    base0D: Color
    base0E: Color
    base0F: Color


@dataclass
class AppTheme:
    name: str
    author: str

    window_bg: Color
    panel_bg: Color
    panel_border: Color
    panel_border_active: Color

    text_primary: Color
    text_muted: Color
    text_accent: Color
    text_error: Color
    text_warning: Color

    font: str
    mono_font: str | None
    font_small_size: int
    font_main_size: int
    font_heading_size: int
    font_title_size: int

    @classmethod
    def from_base16(
        cls,
        p: Base16Palette,
        font: str = DEFAULT_FONT,
        mono_font: str | None = DEFAULT_MONO_FONT,
        font_small_size: int = DEFAULT_FONT_SMALL_SIZE,
        font_main_size: int = DEFAULT_FONT_MAIN_SIZE,
        font_heading_size: int = DEFAULT_FONT_HEADING_SIZE,
        font_title_size: int = DEFAULT_FONT_TITLE_SIZE,
    ) -> "AppTheme":
        return cls(
            name=p.scheme,
            author=p.author,
            font=font,
            mono_font=mono_font,
            font_small_size=font_small_size,
            font_main_size=font_main_size,
            font_heading_size=font_heading_size,
            font_title_size=font_title_size,
            window_bg=p.base00,
            panel_bg=p.base01,
            panel_border=p.base03,
            panel_border_active=p.base0D,
            text_primary=p.base05,
            text_muted=p.base04,
            text_accent=p.base0D,
            text_error=p.base08,
            text_warning=p.base0A,
        )


DEFAULT_PALETTE = Base16Palette(
    scheme="Default Scheme",
    author="KIT",
    base00=Color.from_hex("#1a1b26"),
    base01=Color.from_hex("#1f2335"),
    base02=Color.from_hex("#24283b"),
    base03=Color.from_hex("#414868"),
    base04=Color.from_hex("#565f89"),
    base05=Color.from_hex("#c0caf5"),
    base06=Color.from_hex("#a9b1d6"),
    base07=Color.from_hex("#c0caf5"),
    base08=Color.from_hex("#f7768e"),
    base09=Color.from_hex("#ff9e64"),
    base0A=Color.from_hex("#e0af68"),
    base0B=Color.from_hex("#9ece6a"),
    base0C=Color.from_hex("#2ac3de"),
    base0D=Color.from_hex("#7aa2f7"),
    base0E=Color.from_hex("#bb9af7"),
    base0F=Color.from_hex("#cfc9c2"),
)

DEFAULT_THEME = AppTheme.from_base16(DEFAULT_PALETTE)
