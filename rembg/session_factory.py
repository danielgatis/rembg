import os
from typing import Optional, Type

import onnxruntime as ort

from .sessions import sessions_class
from .sessions.base import BaseSession
from .sessions.u2net import U2netSession


def new_session(model_name: str = "u2net", *args, **kwargs) -> BaseSession:
    """
    Create a new session object based on the specified model name.

    This function searches for the session class based on the model name in the 'sessions_class' list.
    It then creates an instance of the session class with the provided arguments.
    The 'sess_opts' object is created using the 'ort.SessionOptions()' constructor.
    If the 'OMP_NUM_THREADS' environment variable is set, the 'inter_op_num_threads' option of 'sess_opts' is set to its value.

    Parameters:
        model_name (str): The name of the model.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Raises:
        ValueError: If no session class with the given `model_name` is found.

    Returns:
        BaseSession: The created session object.
    """
    session_class: Optional[Type[BaseSession]] = None

    for sc in sessions_class:
        if sc.name() == model_name:
            session_class = sc
            break

    if session_class is None:
        raise ValueError(f"No session class found for model '{model_name}'")

    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        threads = int(os.environ["OMP_NUM_THREADS"])
        sess_opts.inter_op_num_threads = threads
        sess_opts.intra_op_num_threads = threads

    return session_class(model_name, sess_opts, *args, **kwargs)
