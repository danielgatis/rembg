import os
from typing import Optional, Type

import onnxruntime as ort

from .sessions import sessions_class
from .sessions.base import BaseSession


def new_session(
    model_name: str = "u2net",
    *args,
    sess_opts: Optional[ort.SessionOptions] = None,
    **kwargs
) -> BaseSession:
    """
    Create a new session object based on the specified model name.

    This function searches for the session class based on the model name in the 'sessions_class' list.
    It then creates an instance of the session class with the provided arguments.
    If not provided, the 'sess_opts' object is created using the 'ort.SessionOptions()' constructor.
    If the 'OMP_NUM_THREADS' environment variable is set, the 'inter_op_num_threads' and 'intra_op_num_threads' options of 'sess_opts' are set to its value if they are using the default value of 0.

    Parameters:
        model_name (str): The name of the model.
        sess_opts (Optional[ort.SessionOptions]): The ONNX runtime session options. If not provided, a new instance is created.
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

    if sess_opts is None:
        sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        threads = int(os.environ["OMP_NUM_THREADS"])
        if sess_opts.inter_op_num_threads == 0:
            sess_opts.inter_op_num_threads = threads
        if sess_opts.intra_op_num_threads == 0:
            sess_opts.intra_op_num_threads = threads

    return session_class(model_name, sess_opts, *args, **kwargs)
