from . import BiRefNetSessionGeneral


class BiRefNetSessionCOD(BiRefNetSessionGeneral):
    """
    This class represents a BiRefNet-COD session, which is a subclass of BiRefNetSessionGeneral.
    """

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-COD session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-cod"

    @classmethod
    def url_fname(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-COD file in the model url.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model file in the model url.
        """
        return "BiRefNet-COD-epoch_125.onnx"

    @classmethod
    def model_hash(cls, *args, **kwargs):
        """
        Returns the hash of the BiRefNet-COD model file.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The hash of the model file.
        """
        return "md5:f6d0d21ca89d287f17e7afe9f5fd3b45"
