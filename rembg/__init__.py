import sys

if not (sys.version_info.major == 3 and sys.version_info.minor == 9):
    print("Python 3.9.* is required", file=sys.stderr)
    sys.exit(1)

from . import _version

__version__ = _version.get_versions()["version"]

from .bg import remove
