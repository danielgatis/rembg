import json
import sys
from typing import IO

import click

from ..bg import remove
from ..session_factory import new_session
from ..sessions import sessions_names


@click.command(  # type: ignore
    name="i",
    help="for a file as input",
)
@click.option(
    "-m",
    "--model",
    default="u2net",
    type=click.Choice(sessions_names),
    show_default=True,
    show_choices=True,
    help="model name",
)
@click.option(
    "-a",
    "--alpha-matting",
    is_flag=True,
    show_default=True,
    help="use alpha matting",
)
@click.option(
    "-af",
    "--alpha-matting-foreground-threshold",
    default=240,
    type=int,
    show_default=True,
    help="trimap fg threshold",
)
@click.option(
    "-ab",
    "--alpha-matting-background-threshold",
    default=10,
    type=int,
    show_default=True,
    help="trimap bg threshold",
)
@click.option(
    "-ae",
    "--alpha-matting-erode-size",
    default=10,
    type=int,
    show_default=True,
    help="erode size",
)
@click.option(
    "-om",
    "--only-mask",
    is_flag=True,
    show_default=True,
    help="output only the mask",
)
@click.option(
    "-ppm",
    "--post-process-mask",
    is_flag=True,
    show_default=True,
    help="post process the mask",
)
@click.option(
    "-bgc",
    "--bgcolor",
    default=(0, 0, 0, 0),
    type=(int, int, int, int),
    nargs=4,
    help="Background color (R G B A) to replace the removed background with",
)
@click.option("-x", "--extras", type=str)
@click.argument(
    "input", default=(None if sys.stdin.isatty() else "-"), type=click.File("rb")
)
@click.argument(
    "output",
    default=(None if sys.stdin.isatty() else "-"),
    type=click.File("wb", lazy=True),
)
def i_command(model: str, extras: str, input: IO, output: IO, **kwargs) -> None:
    """
    Click command line interface function to process an input file based on the provided options.

    This function is the entry point for the CLI program. It reads an input file, applies image processing operations based on the provided options, and writes the output to a file.

    Parameters:
        model (str): The name of the model to use for image processing.
        extras (str): Additional options in JSON format.
        input: The input file to process.
        output: The output file to write the processed image to.
        **kwargs: Additional keyword arguments corresponding to the command line options.

    Returns:
        None
    """
    try:
        kwargs.update(json.loads(extras))
    except Exception:
        pass

    output.write(remove(input.read(), session=new_session(model, **kwargs), **kwargs))
