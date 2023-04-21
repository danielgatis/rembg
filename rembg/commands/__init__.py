from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules

command_functions = []

package_dir = Path(__file__).resolve().parent
for _b, module_name, _p in iter_modules([str(package_dir)]):
    module = import_module(f"{__name__}.{module_name}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if attribute_name.endswith("_command"):
            command_functions.append(attribute)
