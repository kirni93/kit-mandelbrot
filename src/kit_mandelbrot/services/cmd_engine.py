from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from kit_mandelbrot.app_context import AppContext

CommandHandler = Callable[[AppContext, list[str]], "CommandResult"]


@dataclass
class Command:
    name: str
    aliases: Sequence[str]
    handler: CommandHandler
    summary: str = ""
    usage: str = ""


@dataclass
class CommandResult:
    message: str | None = None
    error: bool = False


class CommandEngine:
    def __init__(self) -> None:
        self._context: Optional[AppContext] = None
        self._commands: dict[str, Command] = {}

    def mount(self, app_ctx: AppContext) -> None:
        self._context = app_ctx

    def register(self, cmd: Command) -> None:
        self._commands[cmd.name] = cmd
        for name in cmd.aliases:
            self._commands[name] = cmd

    def command_help(self, name: str) -> str:
        cmd = self._commands.get(name)
        return "No such command." if cmd is None else f"{cmd.usage} {cmd.summary}"

    def prompt_suggest(self, line: str) -> list[str]:
        raw = line
        stripped = line.lstrip()

        # return all main commands when no real input
        if not stripped:
            return sorted({cmd.name for cmd in self._commands.values()})

        tokens = stripped.split()
        traling_space = raw.endswith(" ")

        # no command linking for now -> assume only first part is a command
        if len(tokens) == 1 and not traling_space:
            curr = tokens[0]
            return sorted(
                {key for key in self._commands.keys() if key.startswith(curr)}
            )

        return []

    def execute(self, line: str) -> CommandResult:
        if self._context is None:
            return CommandResult(message="No app context mounted.", error=True)

        line = line.strip()

        if not line:
            return CommandResult()

        parts = line.split()

        cmd_name, args = parts[0], parts[1:]

        cmd = self._commands.get(cmd_name)

        if cmd is None:
            return CommandResult(
                message=f"Command {cmd_name} not existing!", error=True
            )

        try:
            return cmd.handler(self._context, args)
        except Exception as e:
            return CommandResult(
                message=f"Command '{cmd_name}' failed: {e}", error=True
            )
