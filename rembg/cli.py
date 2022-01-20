import pathlib
import sys
from enum import Enum
from typing import IO, Optional

import click
import filetype
import onnxruntime as ort
import requests
import uvicorn
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
    "-az",
    "--alpha-matting-base-size",
    default=1000,
    type=int,
    show_default=True,
    help="image base size",
)
@click.option(
    "-w",
    "--width",
    default=None,
    type=int,
    show_default=True,
    help="output image size",
)
@click.option(
    "-h",
    "--height",
    default=None,
    type=int,
    show_default=True,
    help="output image size",
)
@click.argument(
    "input", default=(None if sys.stdin.isatty() else "-"), type=click.File("rb")
)
@click.argument(
    "output",
    default=(None if sys.stdin.isatty() else "-"),
    type=click.File("wb", lazy=True),
)
def i(model: str, input: IO, output: IO, **kwargs: dict):
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
    "-az",
    "--alpha-matting-base-size",
    default=1000,
    type=int,
    show_default=True,
    help="image base size",
)
@click.option(
    "-w",
    "--width",
    default=None,
    type=int,
    show_default=True,
    help="output image size",
)
@click.option(
    "-h",
    "--height",
    default=None,
    type=int,
    show_default=True,
    help="output image size",
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
def p(model: str, input: pathlib.Path, output: pathlib.Path, **kwargs: dict):
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
            az: int = Query(1000, ge=0),
            width: Optional[int] = Query(None, gt=0),
            height: Optional[int] = Query(None, gt=0),
        ):
            self.model = model
            self.width = width
            self.height = height
            self.a = a
            self.af = af
            self.ab = ab
            self.ae = ae
            self.az = az

    def im_without_bg(content: bytes, commons: CommonQueryParams) -> Response:
        return Response(
            remove(
                content,
                session=sessions.setdefault(
                    commons.model.value, ort_session(commons.model.value)
                ),
                width=commons.width,
                height=commons.height,
                alpha_matting=commons.a,
                alpha_matting_foreground_threshold=commons.af,
                alpha_matting_background_threshold=commons.ab,
                alpha_matting_erode_size=commons.ae,
                alpha_matting_base_size=commons.az,
            ),
            media_type="image/png",
        )

    @app.get("/")
    def get_index(url: str, commons: CommonQueryParams = Depends()):
        return im_without_bg(requests.get(url).content, commons)

    @app.post("/")
    def post_index(file: bytes = File(...), commons: CommonQueryParams = Depends()):
        return im_without_bg(file, commons)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level=log_level)


if __name__ == "__main__":
    main()
