import argparse
from enum import Enum
from io import BytesIO
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, Form, Query, UploadFile
from PIL import Image
from starlette.responses import StreamingResponse

from .bg import remove
from .detect import ort_session

sessions = {}
app = FastAPI()


class ModelType(str, Enum):
    u2net = "u2net"
    u2netp = "u2netp"
    u2net_human_seg = "u2net_human_seg"


@app.get("/")
def get_index(
    url: str,
    model: Optional[ModelType] = ModelType.u2net,
    width: Optional[int] = Query(None, gt=0),
    height: Optional[int] = Query(None, gt=0),
    a: Optional[bool] = Query(False),
    af: Optional[int] = Query(240, ge=0),
    ab: Optional[int] = Query(10, ge=0),
    ae: Optional[int] = Query(10, ge=0),
    az: Optional[int] = Query(1000, ge=0),
):
    return StreamingResponse(
        BytesIO(
            remove(
                requests.get(url).content,
                session=sessions.setdefault(model, ort_session(model)),
                width=width,
                height=height,
                alpha_matting=a,
                alpha_matting_foreground_threshold=af,
                alpha_matting_background_threshold=ab,
                alpha_matting_erode_structure_size=ae,
                alpha_matting_base_size=az,
            )
        ),
        media_type="image/png",
    )


@app.post("/")
def post_index(
    file: UploadFile = File(...),
    model: Optional[ModelType] = ModelType.u2net,
    width: Optional[int] = Query(None, gt=0),
    height: Optional[int] = Query(None, gt=0),
    a: Optional[bool] = Query(False),
    af: Optional[int] = Query(240, ge=0),
    ab: Optional[int] = Query(10, ge=0),
    ae: Optional[int] = Query(10, ge=0),
    az: Optional[int] = Query(1000, ge=0),
):
    return StreamingResponse(
        BytesIO(
            remove(
                file.read(),
                session=sessions.setdefault(model, ort_session(model)),
                width=width,
                height=height,
                alpha_matting=a,
                alpha_matting_foreground_threshold=af,
                alpha_matting_background_threshold=ab,
                alpha_matting_erode_structure_size=ae,
                alpha_matting_base_size=az,
            )
        ),
        media_type="image/png",
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-a",
        "--addr",
        default="0.0.0.0",
        type=str,
        help="The IP address to bind to.",
    )

    ap.add_argument(
        "-p",
        "--port",
        default=5000,
        type=int,
        help="The port to bind to.",
    )

    ap.add_argument(
        "-l",
        "--log_level",
        default="info",
        type=str,
        help="The log level.",
    )

    args = ap.parse_args()
    uvicorn.run(
        "rembg.server:app", host=args.addr, port=args.port, log_level=args.log_level
    )


if __name__ == "__main__":
    main()
