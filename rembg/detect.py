import os
import sys

import gdown
import numpy as np
import onnxruntime as ort
from PIL import Image
from skimage import transform


def ort_session(model_name):
    path = os.environ.get(
        "U2NETP_PATH",
        os.path.expanduser(os.path.join("~", ".u2net", model_name + ".onnx")),
    )

    if model_name == "u2netp":
        md5 = "8e83ca70e441ab06c318d82300c84806"
        url = "https://drive.google.com/uc?id=1tNuFmLv0TSNDjYIkjEdeH1IWKQdUA4HR"
    elif model_name == "u2net":
        md5 = "60024c5c889badc19c04ad937298a77b"
        url = "https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab"
    elif model_name == "u2net_human_seg":
        md5 = "c09ddc2e0104f800e3e1bb4652583d1f"
        url = "https://drive.google.com/uc?id=1ZfqwVxu-1XWC1xU1GHIP-FM_Knd_AX5j"
    else:
        assert AssertionError("Choose between u2net, u2netp or u2net_human_seg")

    gdown.cached_download(url, path, md5=md5, quiet=True)
    return ort.InferenceSession(path)


def norm_pred(d):
    ma = np.max(d)
    mi = np.min(d)
    dn = (d - mi) / (ma - mi)

    return dn


def rescale(sample, output_size):
    imidx, image, label = sample["imidx"], sample["image"], sample["label"]

    h, w = image.shape[:2]

    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)

    img = transform.resize(image, (output_size, output_size), mode="constant")
    lbl = transform.resize(
        label,
        (output_size, output_size),
        mode="constant",
        order=0,
        preserve_range=True,
    )

    return {"imidx": imidx, "image": img, "label": lbl}


def color(sample):
    imidx, image, label = sample["imidx"], sample["image"], sample["label"]

    tmpLbl = np.zeros(label.shape)

    if np.max(label) < 1e-6:
        label = label
    else:
        label = label / np.max(label)

    tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
    image = image / np.max(image)
    if image.shape[2] == 1:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
    else:
        tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

    tmpLbl[:, :, 0] = label[:, :, 0]
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpLbl = label.transpose((2, 0, 1))

    return {"imidx": imidx, "image": tmpImg, "label": tmpLbl}


def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    sample = {"imidx": np.array([0]), "image": image, "label": label}
    sample = rescale(sample, 320)
    sample = color(sample)

    return sample


def predict(ort_session, item):
    sample = preprocess(item)
    inputs_test = np.expand_dims(sample["image"], 0).astype(np.float32)

    ort_inputs = {ort_session.get_inputs()[0].name: inputs_test}
    ort_outs = ort_session.run(None, ort_inputs)

    d1 = ort_outs[0]
    pred = d1[:, 0, :, :]
    predict = np.squeeze(norm_pred(pred))
    img = Image.fromarray(predict * 255).convert("RGB")

    return img
