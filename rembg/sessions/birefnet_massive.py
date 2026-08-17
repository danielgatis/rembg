import os

import pooch

from . import BiRefNetSessionGeneral


class BiRefNetSessionMassive(BiRefNetSessionGeneral):
    """
    This class represents a BiRefNet-Massive session, which is a subclass of BiRefNetSessionGeneral.
    """

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Downloads the BiRefNet-Massive model file from a specific URL and saves it.

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
            "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx",
            (
                None
                if cls.checksum_disabled(*args, **kwargs)
                else "md5:33e726a2136a3d59eb0fdf613e31e3e9"
            ),
            fname=fname,
            path=cls.model_dir(*args, **kwargs),
            progressbar=True,
        )

        return os.path.join(cls.model_dir(*args, **kwargs), fname)

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-Massive session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-massive"
