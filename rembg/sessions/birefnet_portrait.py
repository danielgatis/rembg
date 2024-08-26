from . import BiRefNetSessionGeneral


class BiRefNetSessionPortrait(BiRefNetSessionGeneral):
    """
    This class represents a BiRefNet-Portrait session, which is a subclass of BiRefNetSessionGeneral.
    """

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

    @classmethod
    def url_fname(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-Portrait file in the model url.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model file in the model url.
        """
        return "BiRefNet-portrait-epoch_150.onnx"

    @classmethod
    def model_hash(cls, *args, **kwargs):
        """
        Returns the md5 of the BiRefNet-Portrait model file.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The md5 of the model file.
        """
        return "md5:c3a64a6abf20250d090cd055f12a3b67"
