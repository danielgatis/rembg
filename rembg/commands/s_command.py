import json
import os
import webbrowser
from typing import Optional, Tuple, cast

import aiohttp
import click
import gradio as gr
import uvicorn
from asyncer import asyncify
from fastapi import Depends, FastAPI, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from .._version import get_versions
from ..bg import remove
from ..session_factory import new_session
from ..sessions import sessions_names
from ..sessions.base import BaseSession


@click.command(  # type: ignore
    name="s",
    help="for a http server",
)
@click.option(
    "-p",
    "--port",
    default=7000,
    type=int,
    show_default=True,
    help="port",
)
@click.option(
    "-h",
    "--host",
    default="0.0.0.0",
    type=str,
    show_default=True,
    help="host",
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
def s_command(port: int, host: str, log_level: str, threads: int) -> None:
    """
    Command-line interface for running the FastAPI web server.

    This function starts the FastAPI web server with the specified port and log level.
    If the number of worker threads is specified, it sets the thread limiter accordingly.
    """
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
        version=get_versions()["version"],
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
        docs_url="/api",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class CommonQueryParams:
        def __init__(
            self,
            model: str = Query(
                description="Model to use when processing image",
                regex=r"(" + "|".join(sessions_names) + ")",
                default="u2net",
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
            bgc: Optional[str] = Query(default=None, description="Background Color"),
            extras: Optional[str] = Query(
                default=None, description="Extra parameters as JSON"
            ),
        ):
            self.model = model
            self.a = a
            self.af = af
            self.ab = ab
            self.ae = ae
            self.om = om
            self.ppm = ppm
            self.extras = extras
            self.bgc = (
                cast(Tuple[int, int, int, int], tuple(map(int, bgc.split(","))))
                if bgc
                else None
            )

    class CommonQueryPostParams:
        def __init__(
            self,
            model: str = Form(
                description="Model to use when processing image",
                regex=r"(" + "|".join(sessions_names) + ")",
                default="u2net",
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
            bgc: Optional[str] = Query(default=None, description="Background Color"),
            extras: Optional[str] = Query(
                default=None, description="Extra parameters as JSON"
            ),
        ):
            self.model = model
            self.a = a
            self.af = af
            self.ab = ab
            self.ae = ae
            self.om = om
            self.ppm = ppm
            self.extras = extras
            self.bgc = (
                cast(Tuple[int, int, int, int], tuple(map(int, bgc.split(","))))
                if bgc
                else None
            )

    def im_without_bg(content: bytes, commons: CommonQueryParams) -> Response:
        kwargs = {}

        if commons.extras:
            try:
                kwargs.update(json.loads(commons.extras))
            except Exception:
                pass

        return Response(
            remove(
                content,
                session=sessions.setdefault(
                    commons.model, new_session(commons.model, **kwargs)
                ),
                alpha_matting=commons.a,
                alpha_matting_foreground_threshold=commons.af,
                alpha_matting_background_threshold=commons.ab,
                alpha_matting_erode_size=commons.ae,
                only_mask=commons.om,
                post_process_mask=commons.ppm,
                bgcolor=commons.bgc,
                **kwargs,
            ),
            media_type="image/png",
        )

    @app.on_event("startup")
    def startup():
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass

        if threads is not None:
            from anyio import CapacityLimiter
            from anyio.lowlevel import RunVar

            RunVar("_default_thread_limiter").set(CapacityLimiter(threads))

    @app.get(
        path="/api/remove",
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
        path="/api/remove",
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
        return await asyncify(im_without_bg)(file, commons)  # type: ignore

    def gr_app(app):
        def inference(input_path, model, *args):
            output_path = "output.png"
            a, af, ab, ae, om, ppm, cmd_args = args

            kwargs = {
                "alpha_matting": a,
                "alpha_matting_foreground_threshold": af,
                "alpha_matting_background_threshold": ab,
                "alpha_matting_erode_size": ae,
                "only_mask": om,
                "post_process_mask": ppm,
            }

            if cmd_args:
                kwargs.update(json.loads(cmd_args))
            kwargs["session"] = new_session(model, **kwargs)

            with open(input_path, "rb") as i:
                with open(output_path, "wb") as o:
                    input = i.read()
                    output = remove(input, **kwargs)
                    o.write(output)
            return os.path.join(output_path)

        interface = gr.Interface(
            inference,
            [
                gr.components.Image(type="filepath", label="Input"),
                gr.components.Dropdown(sessions_names, value="u2net", label="Models"),
                gr.components.Checkbox(value=True, label="Alpha matting"),
                gr.components.Slider(
                    value=240, minimum=0, maximum=255, label="Foreground threshold"
                ),
                gr.components.Slider(
                    value=10, minimum=0, maximum=255, label="Background threshold"
                ),
                gr.components.Slider(
                    value=40, minimum=0, maximum=255, label="Erosion size"
                ),
                gr.components.Checkbox(value=False, label="Only mask"),
                gr.components.Checkbox(value=True, label="Post process mask"),
                gr.components.Textbox(label="Arguments"),
            ],
            gr.components.Image(type="filepath", label="Output"),
            concurrency_limit=3,
            analytics_enabled=False,
        )

        app = gr.mount_gradio_app(app, interface, path="/")
        return app

    print(
        f"To access the API documentation, go to http://{'localhost' if host == '0.0.0.0' else host}:{port}/api"
    )
    print(
        f"To access the UI, go to http://{'localhost' if host == '0.0.0.0' else host}:{port}"
    )

    uvicorn.run(gr_app(app), host=host, port=port, log_level=log_level)
