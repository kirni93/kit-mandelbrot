from kit_mandelbrot.app_context import AppContext
from kit_mandelbrot.domain.viewport import Viewport
from kit_mandelbrot.services.cmd_engine import Command, CommandResult

VP_USAGE: str = "vp <re_min> <re_max> <im_min> <im_max>"
VP_NAME: str = "vp"
VP_ALIASES: list[str] = ["viewport"]
VP_SUMMARY: str = "Setting the viewport to the provided window."


def cmd_viewport(ctx: AppContext, args: list[str]) -> CommandResult:
    if len(args) != 4:
        return CommandResult(message=f"usage: {VP_USAGE}", error=True)

    try:
        re_min, re_max, im_min, im_max = [float(a) for a in args]
    except ValueError:
        return CommandResult(message="vp: parameters must be numbers", error=True)

    if re_min >= re_max:
        return CommandResult(
            message="vp: re_min must be smaller than re_max", error=True
        )

    if im_min >= im_max:
        return CommandResult(
            message="vp: im_min must be smaller than im_max", error=True
        )

    vp = Viewport(re_min=re_min, re_max=re_max, imag_min=im_min, imag_max=im_max)
    ctx.update_viewport(vp)

    return CommandResult(message=f"vp: updated to {vp}", error=False)


VP_CMD = Command(
    name=VP_NAME,
    aliases=VP_ALIASES,
    handler=cmd_viewport,
    usage=VP_USAGE,
    summary=VP_SUMMARY,
)
