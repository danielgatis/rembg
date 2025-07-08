import os
from typing import Dict, List, Tuple
import warnings

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage


class BaseSession:
    """This is a base class for managing a session with a machine learning model."""

    # Define provider constants
    PROVIDER_TENSORRT = ("TensorrtExecutionProvider", {"trt_engine_cache_enable": True})
    PROVIDER_CUDA = ("CUDAExecutionProvider", {})
    PROVIDER_ROCM = ("ROCMExecutionProvider", {})
    # Define CPU provider
    PROVIDER_CPU = ("CPUExecutionProvider", {})

    # Define GPU provider priority
    DEFAULT_GPU_PROVIDER_PRIORITY = [
        PROVIDER_TENSORRT,
        PROVIDER_CUDA,
        PROVIDER_ROCM
    ]

    def __init__(self, model_name: str, sess_opts: ort.SessionOptions, *args, **kwargs):
        """Initialize an instance of the BaseSession class."""
        self.model_name = model_name

        if "providers" in kwargs and isinstance(kwargs["providers"], list):
            providers = kwargs.pop("providers")
        else:
            providers = self.get_optimal_providers()

        self.inner_session = ort.InferenceSession(
            str(self.__class__.download_models(*args, **kwargs)),
            sess_options=sess_opts,
            providers=providers,
        )

    def get_optimal_providers(self) -> List[Tuple[str, Dict]]:
        """Check if GPU acceleration is available and execute available modes according to
        the priority defined in DEFAULT_GPU_PROVIDER_PRIORITY"""

        """Default providers"""
        providers: List[Tuple[str, Dict]] = [self.PROVIDER_CPU]
        
        """Check if GPU is available first"""
        if ort.get_device().startswith("GPU"):
            available_providers = ort.get_available_providers()
            
            """Check for available GPU providers in priority order"""
            priority_providers = self.DEFAULT_GPU_PROVIDER_PRIORITY
            selected_gpu_provider = None
            
            for provider_name, provider_options in priority_providers:
                if provider_name in available_providers:
                    selected_gpu_provider = provider_name
                    providers.insert(0, (provider_name, provider_options))
                    break
            
            """Warn if GPU detected but no compatible providers found"""
            if selected_gpu_provider is None:
                warnings.warn(
                    f"GPU detected but no compatible providers found. Available providers: {available_providers}. Running in CPU mode only.",
                    RuntimeWarning
                )
        
        return providers

    def normalize(
        self,
        img: PILImage,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
        *args,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / max(np.max(im_ary), 1e-6)

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))

        return {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmpImg, 0)
            .astype(np.float32)
        }

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        raise NotImplementedError

    @classmethod
    def checksum_disabled(cls, *args, **kwargs):
        return os.getenv("MODEL_CHECKSUM_DISABLED", None) is not None

    @classmethod
    def u2net_home(cls, *args, **kwargs):
        return os.path.expanduser(
            os.getenv(
                "U2NET_HOME", os.path.join(os.getenv("XDG_DATA_HOME", "~"), ".u2net")
            )
        )

    @classmethod
    def download_models(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def name(cls, *args, **kwargs):
        raise NotImplementedError
