import click

from ..bg import download_models


@click.command(  # type: ignore
    name="d",
    help="download models",
)
@click.argument("models", nargs=-1)
def d_command(models: tuple[str, ...]) -> None:
    """
    Download models
    """
    download_models(models)
