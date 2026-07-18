import ipaddress
import json
import socket
import webbrowser
from typing import List, Optional, Tuple, Union, cast
from urllib.parse import urljoin, urlparse

import aiohttp
import click
import gradio as gr
import uvicorn
from asyncer import asyncify
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from .. import __version__
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
@click.option(
    "--no-ui",
    is_flag=True,
    default=False,
    show_default=True,
    help="disable the Gradio UI (reduces idle CPU usage)",
)
def s_command(port: int, host: str, log_level: str, threads: int, no_ui: bool) -> None:
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
        version=__version__,
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
        allow_credentials=False,
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

        session = sessions.get(commons.model)
        if session is None:
            session = new_session(commons.model, **kwargs)
            sessions[commons.model] = session

        return Response(
            remove(
                content,
                session=session,
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

    def _is_blocked_ip(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )

    def _resolve_public_ips(host: str) -> list[str]:
        """Resolve a hostname to IPs, rejecting the request if any resolved
        address is private/internal. Returns the safe IPs so the caller can
        pin the connection to them and avoid DNS-rebinding (resolve twice)."""
        try:
            resolved = socket.getaddrinfo(host, None)
        except Exception:
            raise ValueError("Could not resolve hostname.")

        ips: List[str] = []
        for item in resolved:
            addr = str(item[4][0])
            ip = ipaddress.ip_address(addr)
            if _is_blocked_ip(ip):
                raise ValueError(
                    "Requests to private/internal addresses are not allowed."
                )
            ips.append(addr)

        if not ips:
            raise ValueError("Could not resolve hostname.")
        return ips

    def _validate_url(url: str) -> list[str]:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("Only http and https URLs are allowed.")
        if not parsed.hostname:
            raise ValueError("Invalid URL: missing hostname.")
        # If the host is a literal IP, validate it directly; otherwise resolve
        # and validate every address it maps to.
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if _is_blocked_ip(ip):
                raise ValueError(
                    "Requests to private/internal addresses are not allowed."
                )
            return [parsed.hostname]
        except ValueError as e:
            if "does not appear to be" not in str(e):
                raise
        return _resolve_public_ips(parsed.hostname)

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
        try:
            _validate_url(url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        max_bytes = 50 * 1024 * 1024

        # Pin every connection to an already-validated IP so aiohttp cannot
        # re-resolve the hostname to a private address (DNS rebinding).
        class _PinnedResolver(aiohttp.abc.AbstractResolver):
            async def resolve(self, host, port=0, family=socket.AF_INET):
                try:
                    ips = _validate_url(f"http://{host}")
                except ValueError:
                    raise OSError(f"Blocked host: {host}")
                return [
                    {
                        "hostname": host,
                        "host": ip,
                        "port": port,
                        "family": family,
                        "proto": 0,
                        "flags": socket.AI_NUMERICHOST,
                    }
                    for ip in ips
                ]

            async def close(self):
                pass

        connector = aiohttp.TCPConnector(resolver=_PinnedResolver())
        async with aiohttp.ClientSession(connector=connector) as session:
            current_url = url
            for _ in range(5):
                async with session.get(current_url, allow_redirects=False) as response:
                    if response.status in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location")
                        if not location:
                            raise HTTPException(
                                status_code=400,
                                detail="Redirect without Location header.",
                            )
                        current_url = urljoin(current_url, location)
                        try:
                            _validate_url(current_url)
                        except ValueError as e:
                            raise HTTPException(status_code=400, detail=str(e))
                        continue

                    if response.content_length and response.content_length > max_bytes:
                        raise HTTPException(
                            status_code=400, detail="Image exceeds maximum size."
                        )

                    file = await response.content.read(max_bytes + 1)
                    if len(file) > max_bytes:
                        raise HTTPException(
                            status_code=400, detail="Image exceeds maximum size."
                        )
                    return await asyncify(im_without_bg)(file, commons)

            raise HTTPException(status_code=400, detail="Too many redirects.")

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
        def inference(input_image, model, *args):
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

            session = sessions.get(model)
            if session is None:
                session = new_session(model, **kwargs)
                sessions[model] = session
            kwargs["session"] = session

            return remove(input_image, **kwargs)

        interface = gr.Interface(
            inference,
            [
                gr.components.Image(type="pil", label="Input"),
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
            gr.components.Image(type="pil", label="Output"),
            concurrency_limit=3,
            analytics_enabled=False,
        )

        app = gr.mount_gradio_app(app, interface, path="/")
        return app

    print(
        f"To access the API documentation, go to http://{'localhost' if host == '0.0.0.0' else host}:{port}/api"
    )
    if not no_ui:
        print(
            f"To access the UI, go to http://{'localhost' if host == '0.0.0.0' else host}:{port}"
        )

    uvicorn.run(
        app if no_ui else gr_app(app), host=host, port=port, log_level=log_level
    )
