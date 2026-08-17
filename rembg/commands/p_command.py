import json
import pathlib
import sys
import time
from typing import List, cast

import click
import filetype
from tqdm import tqdm
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..bg import remove
from ..session_factory import new_session
from ..sessions import sessions, sessions_names


def _billable_count(inputs: List[pathlib.Path], output: pathlib.Path) -> int:
    """Count inputs that would actually reach the model.

    Mirrors the skips in `process`: directories, non-images, and inputs whose
    output already exists are never sent, so counting the raw glob would
    overstate the cost.
    """
    count = 0
    for each_input in inputs:
        if each_input.is_dir():
            continue
        mimetype = filetype.guess(each_input)
        if mimetype is None or mimetype.mime.find("image") < 0:
            continue
        if (output / each_input.name).with_suffix(".png").exists():
            continue
        count += 1
    return count


def _confirm_usage_cost(
    model: str,
    inputs: List[pathlib.Path],
    output: pathlib.Path,
    watch: bool,
    accept_usage_cost: bool,
) -> None:
    """Warn before a billed batch, and confirm unless the cost was accepted.

    Non-interactive runs must pass --accept-usage-cost. Prompting would hang a
    pipeline, and proceeding on its own would spend the user's money with no
    way to decline.
    """
    if watch:
        click.echo(
            f"Warning: '{model}' bills per image and --watch has no fixed end. "
            "Every file added to the folder will be sent and charged.",
            err=True,
        )
    else:
        count = _billable_count(inputs, output)
        if count == 0:
            return
        click.echo(
            f"Warning: '{model}' bills per image. "
            f"This run will send {count} image{'s' if count != 1 else ''} "
            "to a remote service.",
            err=True,
        )

    if accept_usage_cost:
        return

    if not sys.stdin.isatty():
        click.echo(
            "Refusing to run unattended. Pass --accept-usage-cost to proceed.",
            err=True,
        )
        raise click.Abort()

    if not click.confirm("Continue?", default=False, err=True):
        raise click.Abort()


@click.command(  # type: ignore
    name="p",
    help="for a folder as input",
)
@click.option(
    "-m",
    "--model",
    default="bria-rmbg",
    type=click.Choice(sessions_names),
    show_default=True,
    show_choices=True,
    help="model name",
)
@click.option(
    "-a",
    "--alpha-matting",
    is_flag=True,
    show_default=True,
    help="use alpha matting",
)
@click.option(
    "-af",
    "--alpha-matting-foreground-threshold",
    default=240,
    type=int,
    show_default=True,
    help="trimap fg threshold",
)
@click.option(
    "-ab",
    "--alpha-matting-background-threshold",
    default=10,
    type=int,
    show_default=True,
    help="trimap bg threshold",
)
@click.option(
    "-ae",
    "--alpha-matting-erode-size",
    default=10,
    type=int,
    show_default=True,
    help="erode size",
)
@click.option(
    "-om",
    "--only-mask",
    is_flag=True,
    show_default=True,
    help="output only the mask",
)
@click.option(
    "-ppm",
    "--post-process-mask",
    is_flag=True,
    show_default=True,
    help="post process the mask",
)
@click.option(
    "-dc",
    "--decontaminate",
    is_flag=True,
    show_default=True,
    help="remove the background color fringing left on soft edges",
)
@click.option(
    "-vm",
    "--vitmatte",
    is_flag=True,
    show_default=True,
    help="refine the edges with ViTMatte, recovering more fine detail than -a",
)
@click.option(
    "-w",
    "--watch",
    default=False,
    is_flag=True,
    show_default=True,
    help="watches a folder for changes",
)
@click.option(
    "-d",
    "--delete_input",
    default=False,
    is_flag=True,
    show_default=True,
    help="delete input file after processing",
)
@click.option(
    "--accept-usage-cost",
    default=False,
    is_flag=True,
    show_default=True,
    help="proceed without confirming for models that bill per image",
)
@click.option(
    "-bgc",
    "--bgcolor",
    default=(0, 0, 0, 0),
    type=(int, int, int, int),
    nargs=4,
    help="Background color (R G B A) to replace the removed background with",
)
@click.option("-x", "--extras", type=str)
@click.argument(
    "input",
    type=click.Path(
        exists=True,
        path_type=pathlib.Path,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
)
@click.argument(
    "output",
    type=click.Path(
        exists=False,
        path_type=pathlib.Path,
        file_okay=False,
        dir_okay=True,
        writable=True,
    ),
)
def p_command(
    model: str,
    extras: str,
    input: pathlib.Path,
    output: pathlib.Path,
    watch: bool,
    delete_input: bool,
    accept_usage_cost: bool,
    **kwargs,
) -> None:
    """
    Command-line interface (CLI) program for performing background removal on images in a folder.

    This program takes a folder as input and uses a specified model to remove the background from the images in the folder.
    It provides various options for configuration, such as choosing the model, enabling alpha matting, setting trimap thresholds, erode size, etc.
    Additional options include outputting only the mask and post-processing the mask.
    The program can also watch the input folder for changes and automatically process new images.
    The resulting images with the background removed are saved in the specified output folder.

    Parameters:
        model (str): The name of the model to use for background removal.
        extras (str): Additional options in JSON format.
        input (pathlib.Path): The path to the input folder.
        output (pathlib.Path): The path to the output folder.
        watch (bool): Whether to watch the input folder for changes.
        delete_input (bool): Whether to delete the input file after processing.
        accept_usage_cost (bool): Proceed without confirming for models that bill per image.
        **kwargs: Additional keyword arguments.

    Returns:
        None
    """
    try:
        kwargs.update(json.loads(extras))
    except Exception:
        pass

    session = new_session(model, **kwargs)

    def process(each_input: pathlib.Path) -> None:
        try:
            mimetype = filetype.guess(each_input)
            if mimetype is None:
                return
            if mimetype.mime.find("image") < 0:
                return

            each_output = (output / each_input.name).with_suffix(".png")
            each_output.parents[0].mkdir(parents=True, exist_ok=True)

            if not each_output.exists():
                each_output.write_bytes(
                    cast(
                        bytes,
                        remove(each_input.read_bytes(), session=session, **kwargs),
                    )
                )

                if watch:
                    print(
                        f"processed: {each_input.absolute()} -> {each_output.absolute()}"
                    )

            if delete_input:
                each_input.unlink()

        except Exception as e:
            print(e)

    inputs = list(input.glob("**/*"))

    if sessions[model].has_usage_cost():
        _confirm_usage_cost(model, inputs, output, watch, accept_usage_cost)

    inputs_tqdm = inputs if watch else tqdm(inputs)

    for each_input in inputs_tqdm:
        if not each_input.is_dir():
            process(each_input)

    if watch:
        should_watch = True
        observer = Observer()

        class EventHandler(FileSystemEventHandler):
            def on_any_event(self, event: FileSystemEvent) -> None:
                src_path = cast(str, event.src_path)
                if (
                    not (
                        event.is_directory or event.event_type in ["deleted", "closed"]
                    )
                    and pathlib.Path(src_path).exists()
                ):
                    if src_path.endswith("stop.txt"):
                        nonlocal should_watch
                        should_watch = False
                        pathlib.Path(src_path).unlink()
                        return

                    process(pathlib.Path(src_path))

        event_handler = EventHandler()
        observer.schedule(event_handler, str(input), recursive=False)
        observer.start()

        try:
            while should_watch:
                time.sleep(1)

        finally:
            observer.stop()
            observer.join()
