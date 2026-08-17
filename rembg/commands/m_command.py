import os
import pathlib
import shutil

import click

from ..sessions import sessions_class
from ..sessions.base import BaseSession


def _model_for_file(fname: str) -> str | None:
    """Work out which model a file in the old flat directory belongs to.

    Returns None for anything unrecognised, so migration never guesses.
    """
    names = []
    for session_class in sessions_class:
        try:
            names.append(session_class.name())
        except Exception:
            continue

    # SAM ships several files, all sharing a `sam_` prefix.
    if fname.startswith("sam_"):
        return "sam" if "sam" in names else None

    stem = fname[: -len(".onnx")] if fname.endswith(".onnx") else fname
    return stem if stem in names else None


def _is_orphan(fname: str) -> bool:
    """Whether a file is a leftover from an interrupted download."""
    return fname.startswith("tmp") and "." not in fname


@click.command(  # type: ignore
    name="m",
    help="migrate models from the legacy ~/.u2net directory",
)
@click.option(
    "--dry-run",
    is_flag=True,
    show_default=True,
    help="show what would happen without touching any file",
)
@click.option(
    "--delete-source",
    is_flag=True,
    show_default=True,
    help="remove each file from the legacy directory once it is copied",
)
@click.option(
    "--clean-orphans",
    is_flag=True,
    show_default=True,
    help="also delete leftover tmp files from interrupted downloads",
)
def m_command(dry_run: bool, delete_source: bool, clean_orphans: bool) -> None:
    """
    Move downloaded models from the old flat layout into the per-model one.

    Files are copied, not moved, unless --delete-source is passed: a partial
    migration should never be able to lose a multi-gigabyte download.
    """
    legacy = pathlib.Path(BaseSession.legacy_home())
    target_root = pathlib.Path(BaseSession.rembg_home())

    if legacy == target_root:
        click.echo(
            "Legacy and current directories are the same "
            f"({legacy}); nothing to migrate.",
            err=True,
        )
        return

    if not legacy.is_dir():
        click.echo(f"No legacy directory at {legacy}; nothing to migrate.")
        return

    entries = sorted(p for p in legacy.iterdir() if p.is_file())
    if not entries:
        click.echo(f"Legacy directory {legacy} is empty; nothing to migrate.")
        return

    planned: list[tuple[pathlib.Path, pathlib.Path]] = []
    orphans: list[pathlib.Path] = []
    skipped: list[tuple[pathlib.Path, str]] = []
    reclaimable: list[pathlib.Path] = []

    for src in entries:
        if _is_orphan(src.name):
            orphans.append(src)
            continue

        model = _model_for_file(src.name)
        if model is None:
            skipped.append((src, "unrecognised file"))
            continue

        dst = target_root / "models" / model / src.name
        if dst.exists():
            # Already copied by an earlier run. Only reclaimable if the copy is
            # intact, otherwise leave the original alone.
            if delete_source and dst.stat().st_size == src.stat().st_size:
                reclaimable.append(src)
            else:
                skipped.append((src, "already present in the new layout"))
            continue

        planned.append((src, dst))

    prefix = "[dry-run] " if dry_run else ""

    for src, dst in planned:
        size_mb = src.stat().st_size / (1024 * 1024)
        click.echo(f"{prefix}{src.name} -> {dst.parent} ({size_mb:.1f} MB)")

        if dry_run:
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        # Copy first, then unlink, so an interrupted run cannot lose the file.
        shutil.copy2(src, dst)
        if delete_source:
            os.remove(src)

    for src in reclaimable:
        size_mb = src.stat().st_size / (1024 * 1024)
        click.echo(
            f"{prefix}removing {src.name} from the legacy directory "
            f"(already migrated, {size_mb:.1f} MB)"
        )
        if not dry_run:
            os.remove(src)

    for src, reason in skipped:
        click.echo(f"{prefix}skipped {src.name}: {reason}")

    if orphans:
        total_mb = sum(p.stat().st_size for p in orphans) / (1024 * 1024)
        if clean_orphans:
            for src in orphans:
                click.echo(f"{prefix}removing orphan {src.name}")
                if not dry_run:
                    os.remove(src)
        else:
            click.echo(
                f"{prefix}found {len(orphans)} leftover download file(s) "
                f"totalling {total_mb:.1f} MB; pass --clean-orphans to delete them"
            )

    click.echo(
        f"{prefix}migrated {len(planned)} file(s), "
        f"reclaimed {len(reclaimable)}, "
        f"skipped {len(skipped)}, orphans {len(orphans)}"
    )

    if planned and not delete_source and not dry_run:
        click.echo(
            f"The originals are still in {legacy}. "
            "Re-run with --delete-source once you have verified the new layout."
        )
