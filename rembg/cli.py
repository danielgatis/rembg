import pathlib
import sys
import time
from enum import Enum
from typing import IO, cast

import aiohttp
import click
import filetype
import uvicorn
from asyncer import asyncify
from fastapi import Depends, FastAPI, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from tqdm import tqdm
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from . import _version
from .bg import remove
from .session_base import BaseSession
from .session_factory import new_session


@click.group()
@click.version_option(version=_version.get_versions()["version"])
def main() -> None:
    pass


@main.command(help="for a file as input")
@click.option(
    "-m",
    "--model",
    default="u2net",
    type=click.Choice(
        ["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta"]
    ),
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
@click.argument(
    "input", default=(None if sys.stdin.isatty() else "-"), type=click.File("rb")
)
@click.argument(
    "output",
    default=(None if sys.stdin.isatty() else "-"),
    type=click.File("wb", lazy=True),
)
def i(model: str, input: IO, output: IO, **kwargs) -> None:
    output.write(remove(input.read(), session=new_session(model), **kwargs))


@main.command(help="for a folder as input")
@click.option(
    "-m",
    "--model",
    default="u2net",
    type=click.Choice(
        ["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta"]
    ),
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
    "-w",
    "--watch",
    default=False,
    is_flag=True,
    show_default=True,
    help="watches a folder for changes",
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
def p(
    model: str, input: pathlib.Path, output: pathlib.Path, watch: bool, **kwargs
) -> None:
    session = new_session(model)

    def process(each_input: pathlib.Path) -> None:
        try:
            mimetype = filetype.guess(each_input)
            if mimetype is None:
                return
            if mimetype.mime.find("image") < 0:
                return

            each_output = (output / each_input.name).with_suffix(".png")
            each_output.parents[0].mkdir(parents=True, exist_ok=True)

            if not each_output.exists():
                each_output.write_bytes(
                    cast(
                        bytes,
                        remove(each_input.read_bytes(), session=session, **kwargs),
                    )
                )

                if watch:
                    print(
                        f"processed: {each_input.absolute()} -> {each_output.absolute()}"
                    )
        except Exception as e:
            print(e)

    inputs = list(input.glob("**/*"))
    if not watch:
        inputs = tqdm(inputs)

    for each_input in inputs:
        if not each_input.is_dir():
            process(each_input)

    if watch:
        observer = Observer()

        class EventHandler(FileSystemEventHandler):
            def on_any_event(self, event: FileSystemEvent) -> None:
                if not (
                    event.is_directory or event.event_type in ["deleted", "closed"]
                ):
                    process(pathlib.Path(event.src_path))

        event_handler = EventHandler()
        observer.schedule(event_handler, input, recursive=False)
        observer.start()

        try:
            while True:
                time.sleep(1)

        finally:
            observer.stop()
            observer.join()


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
@click.option(
    "-t",
    "--threads",
    default=None,
    type=int,
    show_default=True,
    help="number of worker threads",
)
def s(port: int, log_level: str, threads: int) -> None:
    sessions: dict[str, BaseSession] = {}
    tags_metadata = [
        {
            "name": "Background Removal",
            "description": "Endpoints that perform background removal with different image sources.",
            "externalDocs": {
                "description": "GitHub Source",
                "url": "https://github.com/danielgatis/rembg",
            },
        },
    ]
    app = FastAPI(
        title="Rembg",
        description="Rembg is a tool to remove images background. That is it.",
        version=_version.get_versions()["version"],
        contact={
            "name": "Daniel Gatis",
            "url": "https://github.com/danielgatis",
            "email": "danielgatis@gmail.com",
        },
        license_info={
            "name": "MIT License",
            "url": "https://github.com/danielgatis/rembg/blob/main/LICENSE.txt",
        },
        openapi_tags=tags_metadata,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class ModelType(str, Enum):
        u2net = "u2net"
        u2netp = "u2netp"
        u2net_human_seg = "u2net_human_seg"
        u2net_cloth_seg = "u2net_cloth_seg"
        silueta = "silueta"

    class CommonQueryParams:
        def __init__(
            self,
            model: ModelType = Query(
                default=ModelType.u2net,
                description="Model to use when processing image",
            ),
            a: bool = Query(default=False, description="Enable Alpha Matting"),
            af: int = Query(
                default=240,
                ge=0,
                le=255,
                description="Alpha Matting (Foreground Threshold)",
            ),
            ab: int = Query(
                default=10,
                ge=0,
                le=255,
                description="Alpha Matting (Background Threshold)",
            ),
            ae: int = Query(
                default=10, ge=0, description="Alpha Matting (Erode Structure Size)"
            ),
            om: bool = Query(default=False, description="Only Mask"),
            ppm: bool = Query(default=False, description="Post Process Mask"),
        ):
            self.model = model
            self.a = a
            self.af = af
            self.ab = ab
            self.ae = ae
            self.om = om
            self.ppm = ppm

    class CommonQueryPostParams:
        def __init__(
            self,
            model: ModelType = Form(
                default=ModelType.u2net,
                description="Model to use when processing image",
            ),
            a: bool = Form(default=False, description="Enable Alpha Matting"),
            af: int = Form(
                default=240,
                ge=0,
                le=255,
                description="Alpha Matting (Foreground Threshold)",
            ),
            ab: int = Form(
                default=10,
                ge=0,
                le=255,
                description="Alpha Matting (Background Threshold)",
            ),
            ae: int = Form(
                default=10, ge=0, description="Alpha Matting (Erode Structure Size)"
            ),
            om: bool = Form(default=False, description="Only Mask"),
            ppm: bool = Form(default=False, description="Post Process Mask"),
        ):
            self.model = model
            self.a = a
            self.af = af
            self.ab = ab
            self.ae = ae
            self.om = om
            self.ppm = ppm

    def im_without_bg(content: bytes, commons: CommonQueryParams) -> Response:
        return Response(
            remove(
                content,
                session=sessions.setdefault(
                    commons.model.value, new_session(commons.model.value)
                ),
                alpha_matting=commons.a,
                alpha_matting_foreground_threshold=commons.af,
                alpha_matting_background_threshold=commons.ab,
                alpha_matting_erode_size=commons.ae,
                only_mask=commons.om,
                post_process_mask=commons.ppm,
            ),
            media_type="image/png",
        )

    @app.on_event("startup")
    def startup():
        if threads is not None:
            from anyio import CapacityLimiter
            from anyio.lowlevel import RunVar

            RunVar("_default_thread_limiter").set(CapacityLimiter(threads))

    @app.get(
        path="/",
        tags=["Background Removal"],
        summary="Remove from URL",
        description="Removes the background from an image obtained by retrieving an URL.",
    )
    async def get_index(
        url: str = Query(
            default=..., description="URL of the image that has to be processed."
        ),
        commons: CommonQueryParams = Depends(),
    ):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                file = await response.read()
                return await asyncify(im_without_bg)(file, commons)

    @app.post(
        path="/",
        tags=["Background Removal"],
        summary="Remove from Stream",
        description="Removes the background from an image sent within the request itself.",
    )
    async def post_index(
        file: bytes = File(
            default=...,
            description="Image file (byte stream) that has to be processed.",
        ),
        commons: CommonQueryPostParams = Depends(),
    ):
        return await asyncify(im_without_bg)(file, commons)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level=log_level)
