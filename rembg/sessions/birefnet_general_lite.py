from . import BiRefNetSessionGeneral


class BiRefNetSessionGeneralLite(BiRefNetSessionGeneral):
    """
    This class represents a BiRefNet-General-Lite session, which is a subclass of BiRefNetSessionGeneral.
    """

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-General-Lite session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-general-lite"

    @classmethod
    def url_fname(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-General-Lite file in the model url.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model file in the model url.
        """
        return "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"

    @classmethod
    def model_md5(cls, *args, **kwargs):
        """
        Returns the md5 of the BiRefNet-General-Lite model file.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The md5 of the model file.
        """
        return "md5:4fab47adc4ff364be1713e97b7e66334"
