import pathlib
import sys
from enum import Enum
from typing import IO, Optional

import aiohttp
import click
import filetype
import onnxruntime as ort
import requests
import uvicorn
from asyncer import asyncify
from fastapi import Depends, FastAPI, File, Query
from starlette.responses import Response
from tqdm import tqdm

from .bg import remove
from .detect import ort_session


@click.group()
@click.version_option()
def main():
    pass


@main.command(help="for a file as input")
@click.option(
    "-m",
    "--model",
    default="u2net",
    type=click.Choice(["u2net", "u2netp", "u2net_human_seg"]),
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
@click.argument(
    "input", default=(None if sys.stdin.isatty() else "-"), type=click.File("rb")
)
@click.argument(
    "output",
    default=(None if sys.stdin.isatty() else "-"),
    type=click.File("wb", lazy=True),
)
def i(model: str, input: IO, output: IO, **kwargs):
    output.write(remove(input.read(), session=ort_session(model), **kwargs))


@main.command(help="for a folder as input")
@click.option(
    "-m",
    "--model",
    default="u2net",
    type=click.Choice(["u2net", "u2netp", "u2net_human_seg"]),
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
@click.argument(
    "input",
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
)
@click.argument(
    "output",
    type=click.Path(
        exists=False,
        path_type=pathlib.Path,
        file_okay=False,
        dir_okay=True,
        writable=True,
    ),
)
def p(model: str, input: pathlib.Path, output: pathlib.Path, **kwargs):
    session = ort_session(model)
    for each_input in tqdm(list(input.glob("**/*"))):
        if each_input.is_dir():
            continue

        mimetype = filetype.guess(each_input)
        if mimetype is None:
            continue
        if mimetype.mime.find("image") < 0:
            continue

        each_output = (output / each_input.name).with_suffix(".png")
        each_output.parents[0].mkdir(parents=True, exist_ok=True)

        each_output.write_bytes(
            remove(each_input.read_bytes(), session=session, **kwargs)
        )


@main.command(help="for a http server")
@click.option(
    "-p",
    "--port",
    default=5000,
    type=int,
    show_default=True,
    help="port",
)
@click.option(
    "-l",
    "--log_level",
    default="info",
    type=str,
    show_default=True,
    help="log level",
)
def s(port: int, log_level: str):
    sessions: dict[str, ort.InferenceSession] = {}
    app = FastAPI()

    class ModelType(str, Enum):
        u2net = "u2net"
        u2netp = "u2netp"
        u2net_human_seg = "u2net_human_seg"

    class CommonQueryParams:
        def __init__(
            self,
            model: ModelType = Query(ModelType.u2net),
            a: bool = Query(False),
            af: int = Query(240, ge=0),
            ab: int = Query(10, ge=0),
            ae: int = Query(10, ge=0),
            om: bool = Query(False),
        ):
            self.model = model
            self.a = a
            self.af = af
            self.ab = ab
            self.ae = ae
            self.om = om

    def im_without_bg(content: bytes, commons: CommonQueryParams) -> Response:
        return Response(
            remove(
                content,
                session=sessions.setdefault(
                    commons.model.value, ort_session(commons.model.value)
                ),
                alpha_matting=commons.a,
                alpha_matting_foreground_threshold=commons.af,
                alpha_matting_background_threshold=commons.ab,
                alpha_matting_erode_size=commons.ae,
                only_mask=commons.om,
            ),
            media_type="image/png",
        )

    @app.get("/")
    async def get_index(url: str, commons: CommonQueryParams = Depends()):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                file = await response.read()
                return await asyncify(im_without_bg)(file, commons)

    @app.post("/")
    async def post_index(
        file: bytes = File(...), commons: CommonQueryParams = Depends()
    ):
        return await asyncify(im_without_bg)(file, commons)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level=log_level)


if __name__ == "__main__":
    main()
