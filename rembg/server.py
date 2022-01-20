import argparse
from enum import Enum
from typing import Optional

import requests
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Query, UploadFile
from PIL import Image
from starlette.responses import Response
import onnxruntime as ort

from .bg import remove
from .detect import ort_session

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
            alpha_matting_erode_structure_size=commons.ae,
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
