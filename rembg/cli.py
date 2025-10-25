import click

from . import __version__
from .commands import command_functions


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    pass


for command in command_functions:
    main.add_command(command)
