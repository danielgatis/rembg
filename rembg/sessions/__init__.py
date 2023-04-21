from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

from .base import BaseSession

sessions_class = []
sessions_names = []

package_dir = Path(__file__).resolve().parent
for _b, module_name, _p in iter_modules([str(package_dir)]):
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if (
            isclass(attribute)
            and issubclass(attribute, BaseSession)
            and attribute != BaseSession
        ):
            sessions_class.append(attribute)
            sessions_names.append(attribute.name())
