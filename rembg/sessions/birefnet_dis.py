from . import BiRefNetSessionGeneral


class BiRefNetSessionDIS(BiRefNetSessionGeneral):
    """
    This class represents a BiRefNet-DIS session, which is a subclass of BiRefNetSessionGeneral.
    """

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-DIS session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-dis"

    @classmethod
    def url_fname(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-DIS file in the model url.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model file in the model url.
        """
        return "BiRefNet-DIS-epoch_590.onnx"

    @classmethod
    def model_hash(cls, *args, **kwargs):
        """
        Returns the md5 of the BiRefNet-DIS model file.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The md5 of the model file.
        """
        return "md5:2d4d44102b446f33a4ebb2e56c051f2b"
