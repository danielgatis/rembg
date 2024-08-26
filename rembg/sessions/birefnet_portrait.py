import os

import pooch

from . import BiRefNetSessionGeneral


class BiRefNetSessionPortrait(BiRefNetSessionGeneral):
    """
    This class represents a BiRefNet-Portrait session, which is a subclass of BiRefNetSessionGeneral.
    """

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Downloads the BiRefNet-Portrait model file from a specific URL and saves it.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path to the downloaded model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        pooch.retrieve(
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx",
            (
                None
                if cls.checksum_disabled(*args, **kwargs)
                else "md5:c3a64a6abf20250d090cd055f12a3b67"
            ),
            fname=fname,
            path=cls.u2net_home(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.u2net_home(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-Portrait session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-portrait"
