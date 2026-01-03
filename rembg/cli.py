import sys

# Fast path for --version (avoid importing heavy dependencies)
if len(sys.argv) == 2 and sys.argv[1] in ("--version", "-V"):
    from importlib.metadata import version

    print(f"rembg, version {version('rembg')}")
    sys.exit(0)

try:
    import click
except ImportError:
    print("The CLI dependencies are not installed.")
    print("Please install rembg with CLI support:")
    print()
    print('    pip install "rembg[cpu,cli]"  # for CPU')
    print('    pip install "rembg[gpu,cli]"  # for NVIDIA/CUDA GPU')
    print()
    print(
        "For more information, see: https://github.com/danielgatis/rembg#installation"
    )
    sys.exit(1)

from . import __version__
from .commands import command_functions


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    pass


for command in command_functions:
    main.add_command(command)
