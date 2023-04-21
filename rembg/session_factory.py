import os
from typing import Type

import onnxruntime as ort

from .sessions import sessions_class
from .sessions.base import BaseSession
from .sessions.u2net import U2netSession


def new_session(model_name: str = "u2net", *args, **kwargs) -> BaseSession:
    session_class: Type[BaseSession] = U2netSession

    for sc in sessions_class:
        if sc.name() == model_name:
            session_class = sc
            break

    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    return session_class(model_name, sess_opts, *args, **kwargs)
