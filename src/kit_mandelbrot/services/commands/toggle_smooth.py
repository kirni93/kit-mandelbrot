from typing import Sequence
from kit_mandelbrot.app_context import AppContext
from kit_mandelbrot.services.cmd_engine import Command, CommandResult

TOOGLE_SMOOTH_USAGE: str = "toggle-smooth"
TOOGLE_SMOOTH_NAME: str = "toggle-smooth"
TOOGLE_SMOOTH_ALIASES: Sequence[str] = ("ts",)
TOOGLE_SMOOTH_SUMMARY: str = "Toggles the smoothing of the stability rendering."


def cmd_quit(ctx: AppContext, args: list[str]) -> CommandResult:
    if len(args) != 0:
        return CommandResult(message=f"usage: {TOOGLE_SMOOTH_USAGE}", error=True)

    new_val = ctx.toggle_smooth()

    return CommandResult(message=f"Toggled the smooth to {new_val}.", error=False)


TOOGLE_SMOOTH_CMD = Command(
    name=TOOGLE_SMOOTH_NAME,
    aliases=TOOGLE_SMOOTH_ALIASES,
    handler=cmd_quit,
    usage=TOOGLE_SMOOTH_USAGE,
    summary=TOOGLE_SMOOTH_SUMMARY,
)
