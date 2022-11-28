from . import _version

__version__ = _version.get_versions()["version"]

from .bg import remove
from .session_factory import new_session
