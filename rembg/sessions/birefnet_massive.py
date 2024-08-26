from . import BiRefNetSessionGeneral


class BiRefNetSessionMassive(BiRefNetSessionGeneral):
    """
    This class represents a BiRefNet-Massive session, which is a subclass of BiRefNetSessionGeneral.
    """

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
    
    @classmethod
    def url_fname(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-Massive file in the model url.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the model file in the model url.
        """
        return "BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx"
    
    @classmethod
    def model_md5(cls, *args, **kwargs):
        """
        Returns the md5 of the BiRefNet-Massive model file.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The md5 of the model file.
        """
        return "md5:33e726a2136a3d59eb0fdf613e31e3e9"
