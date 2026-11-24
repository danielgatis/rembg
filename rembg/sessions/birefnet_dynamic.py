import os

import pooch

from . import BiRefNetSessionGeneral


class BiRefNetSessionDynamic(BiRefNetSessionGeneral):
    """
    This class represents a BiRefNet-Dynamic session, which is a subclass of BiRefNetSessionGeneral.
    """

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Downloads the BiRefNet-Dynamic model file from a specific URL and saves it.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path to the downloaded model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        existing = cls.resolve_existing(fname, *args, **kwargs)
        if existing is not None:
            return existing

        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet_dynamic-1024x1024.onnx",
            (
                None
                if cls.checksum_disabled(*args, **kwargs)
                else "md5:4f9c4b2d2e16fe1788f98184e7f43d2b"
            ),
            fname=fname,
            path=cls.model_dir(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.model_dir(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-Dynamic session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-dynamic"
