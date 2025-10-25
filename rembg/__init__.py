try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("rembg")
    except PackageNotFoundError:
        __version__ = "0.0.0"  # Fallback for development
except ImportError:
    __version__ = "0.0.0"  # Fallback for older Python versions

from .bg import remove
from .session_factory import new_session
