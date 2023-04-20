import hashlib
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Type

import onnxruntime as ort
import pooch

from .session_base import BaseSession
from .session_cloth import ClothSession
from .session_dis import DisSession
from .session_sam import SamSession
from .session_simple import SimpleSession


def download_model(url: str, md5: str, fname: str, path: Path):
    pooch.retrieve(
        url,
        f"md5:{md5}",
        fname=fname,
        path=path,
        progressbar=True,
    )


def new_session(model_name: str = "u2net") -> BaseSession:
    # Define the model path
    u2net_home = os.getenv(
        "U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")
    )

    fname = f"{model_name}.onnx"
    path = Path(u2net_home).expanduser()
    full_path = Path(u2net_home).expanduser() / fname

    session_class: Type[BaseSession]
    md5 = "60024c5c889badc19c04ad937298a77b"
    url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
    session_class = SimpleSession

    if model_name == "u2netp":
        md5 = "8e83ca70e441ab06c318d82300c84806"
        url = (
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
        )
        session_class = SimpleSession
    elif model_name == "u2net_human_seg":
        md5 = "c09ddc2e0104f800e3e1bb4652583d1f"
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx"
        session_class = SimpleSession
    elif model_name == "u2net_cloth_seg":
        md5 = "2434d1f3cb744e0e49386c906e5a08bb"
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx"
        session_class = ClothSession
    elif model_name == "silueta":
        md5 = "55e59e0d8062d2f5d013f4725ee84782"
        url = (
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx"
        )
        session_class = SimpleSession
    elif model_name == "isnet-general-use":
        md5 = "fc16ebd8b0c10d971d3513d564d01e29"
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
        session_class = DisSession
    elif model_name == "SAM":
        path = Path(u2net_home).expanduser()

        fname_encoder = f"{model_name}_encoder.onnx"
        encoder_md5 = "13d97c5c79ab13ef86d67cbde5f1b250"
        encoder_url = "https://github.com/Flippchen/rembg/releases/download/test/vit_b-encoder-quant.onnx"

        fname_decoder = f"{model_name}_decoder.onnx"
        decoder_md5 = "fa3d1c36a3187d3de1c8deebf33dd127"
        decoder_url = "https://github.com/Flippchen/rembg/releases/download/test/vit_b-decoder-quant.onnx"

        download_model(encoder_url, encoder_md5, fname_encoder, path)
        download_model(decoder_url, decoder_md5, fname_decoder, path)

        sess_opts = ort.SessionOptions()

        if "OMP_NUM_THREADS" in os.environ:
            sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

        return SamSession(
            model_name,
            ort.InferenceSession(
                str(path / fname_encoder),
                providers=ort.get_available_providers(),
                sess_options=sess_opts,
            ),
            ort.InferenceSession(
                str(path / fname_decoder),
                providers=ort.get_available_providers(),
                sess_options=sess_opts,
            ),
        )

    download_model(url, md5, fname, path)

    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    return session_class(
        model_name,
        ort.InferenceSession(
            str(full_path),
            providers=ort.get_available_providers(),
            sess_options=sess_opts,
        ),
    )
