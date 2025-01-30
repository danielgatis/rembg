import asyncio
import io
import json
import os
import sys
from typing import IO

import click
import PIL

from ..bg import remove
from ..session_factory import new_session
from ..sessions import sessions_names


@click.command(  # type: ignore
    name="b",
    help="for a byte stream as input",
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
@click.option(
    "-o",
    "--output_specifier",
    type=str,
    help="printf-style specifier for output filenames (e.g. 'output-%d.png'))",
)
@click.argument(
    "image_width",
    type=int,
)
@click.argument(
    "image_height",
    type=int,
)
def b_command(
    model: str,
    extras: str,
    image_width: int,
    image_height: int,
    output_specifier: str,
    **kwargs
) -> None:
    """
    Command-line interface for processing images by removing the background using a specified model and generating a mask.

    This CLI command takes several options and arguments to configure the background removal process and save the processed images.

    Parameters:
        model (str): The name of the model to use for background removal.
        extras (str): Additional options in JSON format that can be passed to customize the background removal process.
        image_width (int): The width of the input images in pixels.
        image_height (int): The height of the input images in pixels.
        output_specifier (str): A printf-style specifier for the output filenames. If specified, the processed images will be saved to the specified output directory with filenames generated using the specifier.
        **kwargs: Additional keyword arguments that can be used to customize the background removal process.

    Returns:
        None
    """
    if extras:
        try:
            kwargs.update(json.loads(extras))
        except Exception:
            raise click.BadParameter("extras must be a valid JSON string")

    session = new_session(model, **kwargs)
    bytes_per_img = image_width * image_height * 3

    if output_specifier:
        output_dir = os.path.dirname(
            os.path.abspath(os.path.expanduser(output_specifier))
        )

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    def img_to_byte_array(img: PIL.Image.Image) -> bytes:
        buff = io.BytesIO()
        img.save(buff, format="PNG")
        return buff.getvalue()

    async def connect_stdin_stdout():
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)

        await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        w_transport, w_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )

        writer = asyncio.StreamWriter(w_transport, w_protocol, reader, loop)
        return reader, writer

    async def main():
        reader, writer = await connect_stdin_stdout()

        idx = 0
        while True:
            try:
                img_bytes = await reader.readexactly(bytes_per_img)
                if not img_bytes:
                    break

                img = PIL.Image.frombytes("RGB", (image_width, image_height), img_bytes)
                output = remove(img, session=session, **kwargs)

                if output_specifier:
                    output.save((output_specifier % idx), format="PNG")
                else:
                    writer.write(img_to_byte_array(output))

                idx += 1
            except asyncio.IncompleteReadError:
                break

    asyncio.run(main())
