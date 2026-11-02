import importlib
import io
import pathlib

import pytest
from click.testing import CliRunner
from PIL import Image
from watchdog.events import (
    FileClosedNoWriteEvent,
    FileCreatedEvent,
    FileModifiedEvent,
    FileOpenedEvent,
)

from rembg.commands.p_command import p_command

# rembg.commands re-exports the command under the module's own name, so fetch
# the module itself to patch the names it looks up.
p_module = importlib.import_module("rembg.commands.p_command")


def jpeg(color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def processed(monkeypatch):
    """Replace the model with a stub and record every input it is given."""
    calls = []

    def fake_remove(data, *args, **kwargs):
        calls.append(data)
        return b"cutout"

    monkeypatch.setattr(p_module, "new_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(p_module, "remove", fake_remove)
    return calls


@pytest.fixture
def watch_events(monkeypatch):
    """Stand in for watchdog's Observer: once the watch starts, deliver these
    events to the handler, then stop the watch with stop.txt."""
    events = []

    class Observer:
        def schedule(self, handler, path, recursive=False):
            self.handler, self.path = handler, pathlib.Path(path)

        def start(self):
            stop = self.path / "stop.txt"
            stop.write_text("stop")
            for event in [*events, FileCreatedEvent(str(stop))]:
                self.handler.dispatch(event)

        def stop(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(p_module, "Observer", Observer)
    return events


def run(*args):
    return CliRunner().invoke(p_command, ["-m", "u2netp", *map(str, args)])


def test_delete_input_removes_processed_files(tmp_path, processed):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    (src / "photo.jpg").write_bytes(jpeg("red"))

    result = run("-d", src, dst)

    assert result.exit_code == 0
    assert (dst / "photo.png").read_bytes() == b"cutout"
    assert not (src / "photo.jpg").exists()


def test_delete_input_keeps_files_whose_output_name_is_taken(tmp_path, processed):
    """in/a/photo.jpg and in/b/photo.jpg both map to out/photo.png.

    Only one of them is processed, so the other one must not be deleted.
    """
    src, dst = tmp_path / "in", tmp_path / "out"
    for folder, color in (("a", "red"), ("b", "blue")):
        (src / folder).mkdir(parents=True)
        (src / folder / "photo.jpg").write_bytes(jpeg(color))

    result = run("-d", src, dst)

    assert result.exit_code == 0
    assert len(processed) == 1
    kept = list(src.glob("*/photo.jpg"))
    assert len(kept) == 1
    assert kept[0].read_bytes() != processed[0]
    assert "input kept" in result.output


def test_delete_input_keeps_file_when_output_already_exists(tmp_path, processed):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    dst.mkdir()
    (src / "photo.jpg").write_bytes(jpeg("red"))
    (dst / "photo.png").write_bytes(b"earlier run")

    result = run("-d", src, dst)

    assert result.exit_code == 0
    assert processed == []
    assert (src / "photo.jpg").exists()
    assert (dst / "photo.png").read_bytes() == b"earlier run"


def test_existing_output_is_skipped_quietly_without_delete_input(tmp_path, processed):
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    dst.mkdir()
    (src / "photo.jpg").write_bytes(jpeg("red"))
    (dst / "photo.png").write_bytes(b"earlier run")

    result = run(src, dst)

    assert result.exit_code == 0
    assert processed == []
    assert (src / "photo.jpg").exists()
    assert "input kept" not in result.output


def test_watch_ignores_events_from_reading_a_kept_input(
    tmp_path, processed, watch_events
):
    # On Linux, opening the input to check it is reported back as "opened"
    # and "closed_no_write" events for the same file.
    src, dst = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    dst.mkdir()
    photo = src / "photo.jpg"
    photo.write_bytes(jpeg("red"))
    (dst / "photo.png").write_bytes(b"earlier run")
    watch_events.extend(
        [
            FileOpenedEvent(str(photo)),
            FileClosedNoWriteEvent(str(photo)),
            FileModifiedEvent(str(photo)),
        ]
    )

    result = run("-w", "-d", src, dst)

    assert result.exit_code == 0
    assert photo.exists()
    # Once for the initial pass and once for the modification.
    assert result.output.count("input kept") == 2
