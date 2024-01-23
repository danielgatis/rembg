import click

from ..bg import download_models


@click.command(  # type: ignore
    name="d",
    help="download all models",
)
def d_command(*args, **kwargs) -> None:
    """
    Download all models
    """
    download_models()
