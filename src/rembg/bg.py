import argparse
import io
import os

import numpy as np
from PIL import Image

from .u2net import detect

model_u2net = detect.load_model(model_name="u2net")
model_u2netp = detect.load_model(model_name="u2netp")


def remove(data, model_name="u2net"):
    model = model_u2net

    if model == "u2netp":
        model = model_u2netp

    img = Image.open(io.BytesIO(data))
    roi = detect.predict(model, np.array(img))
    roi = roi.resize((img.size), resample=Image.LANCZOS)

    empty = Image.new("RGBA", (img.size), 0)
    out = Image.composite(img, empty, roi.convert("L"))

    bio = io.BytesIO()
    out.save(bio, "PNG")

    return bio.getbuffer()
