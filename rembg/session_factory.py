import hashlib
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Type

import gdown
import onnxruntime as ort

from .session_base import BaseSession
from .session_cloth import ClothSession
from .session_simple import SimpleSession


def new_session(model_name: str) -> BaseSession:
    session_class: Type[BaseSession]

    mean, std, model_size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (320, 320)
    path = None
    sess_opts = ort.SessionOptions()
    if "OMP_NUM_THREADS" in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    if model_name == "u2netp":
        md5 = "8e83ca70e441ab06c318d82300c84806"
        url = "https://drive.google.com/uc?id=1tNuFmLv0TSNDjYIkjEdeH1IWKQdUA4HR"
        session_class = SimpleSession
    elif model_name == "u2net":
        md5 = "60024c5c889badc19c04ad937298a77b"
        url = "https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab"
        session_class = SimpleSession
    elif model_name == "u2net_human_seg":
        md5 = "c09ddc2e0104f800e3e1bb4652583d1f"
        url = "https://drive.google.com/uc?id=1ZfqwVxu-1XWC1xU1GHIP-FM_Knd_AX5j"
        session_class = SimpleSession
    elif model_name == "u2net_cloth_seg":
        md5 = "2434d1f3cb744e0e49386c906e5a08bb"
        url = "https://drive.google.com/uc?id=15rKbQSXQzrKCQurUjZFg8HqzZad8bcyz"
        session_class = ClothSession
        mean, std, model_size = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (768, 768)
    elif model_name.endswith(".onnx"):
        session_class = SimpleSession
        path = model_name
        if "isnet" in model_name:
            model_name = "isnet"
            mean, std, model_size = (0.5, 0.5, 0.5), (1, 1, 1), (1024, 1024)
        else:
            model_name = "u2net"
    else:
        assert AssertionError("Choose between u2net, u2netp, u2net_human_seg or u2net_cloth_seg or path.onnx")
    
    if path is None:
        home = os.getenv("U2NET_HOME", os.path.join("~", ".u2net"))
        path = Path(home).expanduser() / f"{model_name}.onnx"
        path.parents[0].mkdir(parents=True, exist_ok=True)

        if not path.exists():
            with redirect_stdout(sys.stderr):
                gdown.download(url, str(path), use_cookies=False)
        else:
            hashing = hashlib.new("md5", path.read_bytes(), usedforsecurity=False)
            if hashing.hexdigest() != md5:
                with redirect_stdout(sys.stderr):
                    gdown.download(url, str(path), use_cookies=False)
 
    return session_class(
        model_name,
        ort.InferenceSession(str(path), providers=ort.get_available_providers(), sess_options=sess_opts),
        mean=mean,
        std=std,
        size=model_size,
    )
