import os

from .birefnet_general import BiRefNetSessionGeneral


class BiRefNetSessionGeneralFP16(BiRefNetSessionGeneral):
    """
    BiRefNet-General with its weights and activations in float16, for GPUs.

    The model is not downloaded separately: the float32 birefnet-general model is
    fetched (and checksummed) as usual, then converted once and cached. Inputs and
    outputs stay float32, so prediction is identical to BiRefNetSessionGeneral.

    On CUDA this is roughly 30% faster with masks that match float32 to an IoU of
    ~0.9996. It does not reduce VRAM, which is dominated by activations at
    1024x1024. On CPU it is usually slower than float32, so use birefnet-general there.
    """

    @classmethod
    def download_models(cls, *args, **kwargs):
        """
        Returns the path to the float16 model, converting it on first use.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The path to the float16 model file.
        """
        fname = f"{cls.name(*args, **kwargs)}.onnx"
        existing = cls.resolve_existing(fname, *args, **kwargs)
        if existing is not None:
            return existing

        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError as e:
            raise ImportError(
                "birefnet-general-fp16 converts the float32 model on first use and "
                "needs onnx and onnxconverter-common: pip install 'rembg[fp16]'"
            ) from e

        source = BiRefNetSessionGeneral.download_models(*args, **kwargs)
        model = float16.convert_float_to_float16(onnx.load(source), keep_io_types=True)

        model_dir = cls.model_dir(*args, **kwargs)
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, fname)

        # Write then rename, so an interrupted conversion never leaves a partial
        # file that resolve_existing would pick up next time.
        tmp = f"{path}.tmp"
        onnx.save(model, tmp)
        os.replace(tmp, path)

        return path

    @classmethod
    def name(cls, *args, **kwargs):
        """
        Returns the name of the BiRefNet-General-FP16 session.

        Parameters:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The name of the session.
        """
        return "birefnet-general-fp16"
