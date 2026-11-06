import ipaddress
import json
import os
import re
import socket
import threading
import webbrowser
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import aiohttp
import click
import uvicorn
from asyncer import asyncify
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from .. import __version__
from ..bg import remove
from ..session_factory import new_session
from ..sessions import sessions_names, sessions_names_no_credentials
from ..sessions.base import BaseSession

_PROXY_ENV_VARS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)


def _fix_bare_socks_proxy_scheme() -> None:
    """Rewrite a bare `socks://` proxy env var to `socks5://`.

    httpx (pulled in by gradio) only accepts `socks5`/`socks5h` proxy
    schemes and raises `ValueError: Unknown scheme for proxy URL` on
    `socks://`, which crashes `import gradio` before the server even
    starts. Many proxy tools (e.g. Clash) export `socks://` by default,
    so normalize it instead of letting the import blow up.
    """
    for name in _PROXY_ENV_VARS:
        value = os.environ.get(name)
        if value and value.startswith("socks://"):
            os.environ[name] = "socks5://" + value[len("socks://") :]


def _unwrap_ipv6(
    ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address],
) -> Union[ipaddress.IPv4Address, ipaddress.IPv6Address]:
    """Return the IPv4 address embedded in an IPv6 transition address.

    6to4 (`2002::/16`), NAT64 (`64:ff9b::/96`), Teredo and IPv4-mapped
    addresses all carry an IPv4 address inside them. Whether the wrapper
    itself is classified as private depends on the interpreter: on
    Python 3.11.9, `2002:c0a8:0101::` reports `is_private=False` even though
    it embeds 192.168.1.1, so checking only the outer address lets a request
    through to an internal host. Unwrap first, then classify.
    """
    if not isinstance(ip, ipaddress.IPv6Address):
        return ip

    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    if ip.teredo is not None:
        # (server, client); the client is the address traffic reaches.
        return ip.teredo[1]
    if ip in ipaddress.ip_network("64:ff9b::/96"):
        return ipaddress.ip_address(int(ip) & 0xFFFFFFFF)

    return ip


def _is_blocked_ip(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> bool:
    def blocked(addr) -> bool:
        return (
            addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
        )

    # Both the wrapper and what it embeds have to be public. Note that some
    # interpreters already classify whole transition ranges (6to4, NAT64) as
    # private, so those are refused outright rather than by their payload --
    # acceptable, as 6to4 is deprecated (RFC 7526) and NAT64 is operator-local.
    return blocked(ip) or blocked(_unwrap_ipv6(ip))


# Extras a remote caller may pass. Anything else (`providers`, `sess_opts`,
# `session`, ...) reaches internals the caller has no business choosing.
# Session extras pick which session is built, so they are part of its cache key;
# predict extras only steer a single call and must not build a new session.
_SESSION_EXTRAS = frozenset(
    {
        "model_path",
        "sam_model",
        "sam_quant",
        "api_key",
        "model",
        "timeout",
        "poll_timeout",
    }
)
_PREDICT_EXTRAS = frozenset(
    {"sam_prompt", "cc", "cloth_category", "vitmatte_model", "putalpha"}
)


def _parse_extras(raw: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split the caller's JSON extras into (session kwargs, predict kwargs).

    Raises ValueError on malformed JSON or on a key outside the allowlist.
    """
    if not raw:
        return {}, {}

    try:
        extras = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("extras must be valid JSON.")

    if not isinstance(extras, dict):
        raise ValueError("extras must be a JSON object.")

    unknown = set(extras) - _SESSION_EXTRAS - _PREDICT_EXTRAS
    if unknown:
        raise ValueError(
            f"Unsupported extras: {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(_SESSION_EXTRAS | _PREDICT_EXTRAS))}."
        )

    session_kwargs = {k: v for k, v in extras.items() if k in _SESSION_EXTRAS}
    predict_kwargs = {k: v for k, v in extras.items() if k in _PREDICT_EXTRAS}
    return session_kwargs, predict_kwargs


def _parse_bgcolor(raw: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    """Parse `r,g,b` or `r,g,b,a` into RGBA, raising ValueError on anything else.

    A missing alpha means opaque, as Pillow does for a 3-tuple RGBA color.
    """
    if not raw:
        return None

    try:
        color = tuple(int(part) for part in raw.split(","))
    except ValueError:
        raise ValueError("bgc must be comma separated integers, e.g. 255,255,255,255.")

    if len(color) not in (3, 4) or not all(0 <= c <= 255 for c in color):
        raise ValueError("bgc must have 3 or 4 values between 0 and 255.")

    r, g, b, *alpha = color
    return (r, g, b, alpha[0] if alpha else 255)


class _SessionCache:
    """A bounded LRU of sessions.

    Every distinct set of session extras builds a new model in memory, so an
    unbounded cache lets any caller exhaust RAM by varying them.
    """

    def __init__(self, max_size: int):
        self.max_size = max(1, max_size)
        self._sessions: "OrderedDict[tuple, BaseSession]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, model: str, session_kwargs: Dict[str, Any]) -> BaseSession:
        key = (model, json.dumps(session_kwargs, sort_keys=True, default=str))

        with self._lock:
            session = self._sessions.get(key)
            if session is not None:
                self._sessions.move_to_end(key)
                return session

        # Built outside the lock: loading a model can take seconds and must not
        # stall requests for sessions that are already cached.
        session = new_session(model, **session_kwargs)

        with self._lock:
            self._sessions[key] = session
            self._sessions.move_to_end(key)
            while len(self._sessions) > self.max_size:
                self._sessions.popitem(last=False)

        return session


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
@click.option(
    "--allow-remote-models",
    is_flag=True,
    default=False,
    show_default=True,
    help="also serve cloud models (e.g. atlascloud, withoutbg); requests without "
    "their own api_key are billed to this server's API key environment variables",
)
@click.option(
    "--max-sessions",
    default=8,
    type=click.IntRange(min=1),
    show_default=True,
    help="maximum number of models kept loaded in memory",
)
def s_command(
    port: int,
    host: str,
    log_level: str,
    threads: int,
    no_ui: bool,
    allow_remote_models: bool,
    max_sessions: int,
) -> None:
    """
    Command-line interface for running the FastAPI web server.

    This function starts the FastAPI web server with the specified port and log level.
    If the number of worker threads is specified, it sets the thread limiter accordingly.
    """
    sessions = _SessionCache(max_sessions)

    # Cloud models fall back to the API key in this server's environment, so
    # serving them by default would let anyone reaching the port spend it.
    model_names = (
        sessions_names if allow_remote_models else sessions_names_no_credentials
    )
    model_regex = r"^(" + "|".join(re.escape(n) for n in model_names) + r")$"
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
                regex=model_regex,
                default="bria-rmbg",
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
                default=10,
                ge=0,
                le=255,
                description="Alpha Matting (Erode Structure Size)",
            ),
            om: bool = Query(default=False, description="Only Mask"),
            ppm: bool = Query(default=False, description="Post Process Mask"),
            dc: bool = Query(default=False, description="Decontaminate Edges"),
            vm: bool = Query(default=False, description="Refine Edges with ViTMatte"),
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
            self.dc = dc
            self.vm = vm
            self.extras = extras
            try:
                self.bgc = _parse_bgcolor(bgc)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

    class CommonQueryPostParams:
        def __init__(
            self,
            model: str = Form(
                description="Model to use when processing image",
                regex=model_regex,
                default="bria-rmbg",
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
                default=10,
                ge=0,
                le=255,
                description="Alpha Matting (Erode Structure Size)",
            ),
            om: bool = Form(default=False, description="Only Mask"),
            ppm: bool = Form(default=False, description="Post Process Mask"),
            dc: bool = Form(default=False, description="Decontaminate Edges"),
            vm: bool = Form(default=False, description="Refine Edges with ViTMatte"),
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
            self.dc = dc
            self.vm = vm
            self.extras = extras
            try:
                self.bgc = _parse_bgcolor(bgc)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

    def im_without_bg(content: bytes, commons: CommonQueryParams) -> Response:
        try:
            session_kwargs, predict_kwargs = _parse_extras(commons.extras)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Session extras are part of the key: a session built with one caller's
        # api_key or model_path must not be handed to a request that passed another.
        session = sessions.get(commons.model, session_kwargs)
        kwargs = {**session_kwargs, **predict_kwargs}

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
                decontaminate=commons.dc,
                vitmatte=commons.vm,
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
        _fix_bare_socks_proxy_scheme()
        import gradio as gr

        def inference(input_image, model, *args):
            a, af, ab, ae, om, ppm, dc, vm, cmd_args = args

            kwargs = {
                "alpha_matting": a,
                "alpha_matting_foreground_threshold": af,
                "alpha_matting_background_threshold": ab,
                "alpha_matting_erode_size": ae,
                "only_mask": om,
                "post_process_mask": ppm,
                "decontaminate": dc,
                "vitmatte": vm,
            }

            # The dropdown is only a hint: Gradio's API accepts any value here.
            if model not in model_names:
                raise gr.Error(f"Unsupported model: {model}")

            try:
                session_kwargs, predict_kwargs = _parse_extras(cmd_args)
            except ValueError as e:
                raise gr.Error(str(e))

            kwargs.update(session_kwargs)
            kwargs.update(predict_kwargs)
            kwargs["session"] = sessions.get(model, session_kwargs)

            return remove(input_image, **kwargs)

        interface = gr.Interface(
            inference,
            [
                gr.components.Image(type="pil", label="Input"),
                gr.components.Dropdown(model_names, value="bria-rmbg", label="Models"),
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
                gr.components.Checkbox(value=False, label="Decontaminate edges"),
                gr.components.Checkbox(value=False, label="ViTMatte edge refinement"),
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
