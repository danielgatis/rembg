from . import BiRefNetSessionGeneral


class BiRefNetSessionHRSOD(BiRefNetSessionGeneral):
    """
    This class represents a BiRefNet-HRSOD session, which is a subclass of BiRefNetSessionGeneral.
    """

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-HRSOD session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-hrsod"

    @classmethod
    def url_fname(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-HRSOD file in the model url.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model file in the model url.
        """
        return "BiRefNet-HRSOD_DHU-epoch_115.onnx"

    @classmethod
    def model_hash(cls, *args, **kwargs):
        """
        Returns the md5 of the BiRefNet-HRSOD model file.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The md5 of the model file.
        """
        return "md5:c017ade5de8a50ff0fd74d790d268dda"
