import pkg_resources


def main() -> None:
    package_distribution = pkg_resources.get_distribution("rembg")

    for extra in package_distribution.extras:
        if extra == "cli":
            requirements = package_distribution.requires(extras=(extra,))
            for requirement in requirements:
                try:
                    pkg_resources.require(requirement.name)
                except pkg_resources.DistributionNotFound:
                    print(f"Missing dependency: '{requirement.name}'")
                    print(
                        f"Please install rembg with the cli extra: pip install rembg[cli]"
                    )
                    exit(1)

    import click

    from . import _version
    from .commands import command_functions

    @click.group()
    @click.version_option(version=_version.get_versions()["version"])
    def _main() -> None:
        pass

    for command in command_functions:
        _main.add_command(command)

    _main()
