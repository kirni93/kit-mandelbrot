from typing import Sequence
from kit_mandelbrot.app_context import AppContext
from kit_mandelbrot.services.cmd_engine import Command, CommandResult

Q_USAGE: str = "q"
Q_NAME: str = "q"
Q_ALIASES: Sequence[str] = ("quit",)
Q_SUMMARY: str = "Exits the application."


def cmd_quit(ctx: AppContext, args: list[str]) -> CommandResult:
    if len(args) != 0:
        return CommandResult(message=f"usage: {Q_USAGE}", error=True)

    ctx.quit()

    return CommandResult(message="A dopo. 🤌", error=False)


Q_CMD = Command(
    name=Q_NAME,
    aliases=Q_ALIASES,
    handler=cmd_quit,
    usage=Q_USAGE,
    summary=Q_SUMMARY,
)
