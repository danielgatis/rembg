import pytest
from click.testing import CliRunner

from rembg.commands.m_command import m_command


@pytest.fixture
def layout(monkeypatch, tmp_path):
    """A legacy directory with a few files, and an empty destination."""
    monkeypatch.delenv("U2NET_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "legacy_base"))
    monkeypatch.setenv("REMBG_HOME", str(tmp_path / "new"))

    legacy = tmp_path / "legacy_base" / ".u2net"
    legacy.mkdir(parents=True)

    (legacy / "u2net.onnx").write_text("u2net")
    (legacy / "silueta.onnx").write_text("silueta")
    (legacy / "sam_vit_b_01ec64.encoder.onnx").write_text("enc")
    (legacy / "sam_vit_b_01ec64.decoder.onnx").write_text("dec")
    (legacy / "unknown-thing.onnx").write_text("mystery")
    (legacy / "tmpABC123").write_text("interrupted download")

    return {"legacy": legacy, "new": tmp_path / "new"}


def test_dry_run_touches_nothing(layout):
    result = CliRunner().invoke(m_command, ["--dry-run"])

    assert result.exit_code == 0
    assert not layout["new"].exists()
    assert len(list(layout["legacy"].iterdir())) == 6


def test_migrate_copies_into_per_model_dirs(layout):
    result = CliRunner().invoke(m_command, [])

    assert result.exit_code == 0
    assert (layout["new"] / "models" / "u2net" / "u2net.onnx").read_text() == "u2net"
    assert (layout["new"] / "models" / "silueta" / "silueta.onnx").exists()

    # Both SAM files belong to the same model directory.
    sam_dir = layout["new"] / "models" / "sam"
    assert (sam_dir / "sam_vit_b_01ec64.encoder.onnx").exists()
    assert (sam_dir / "sam_vit_b_01ec64.decoder.onnx").exists()


def test_migrate_keeps_originals_by_default(layout):
    CliRunner().invoke(m_command, [])

    # Losing a multi-gigabyte download to a half-finished run is unacceptable.
    assert (layout["legacy"] / "u2net.onnx").exists()


def test_unrecognised_files_are_never_moved(layout):
    CliRunner().invoke(m_command, ["--delete-source", "--clean-orphans"])

    assert (layout["legacy"] / "unknown-thing.onnx").exists()
    assert not (layout["new"] / "models" / "unknown-thing").exists()


def test_orphans_are_kept_unless_asked(layout):
    CliRunner().invoke(m_command, [])
    assert (layout["legacy"] / "tmpABC123").exists()


def test_clean_orphans_removes_them(layout):
    CliRunner().invoke(m_command, ["--clean-orphans"])
    assert not (layout["legacy"] / "tmpABC123").exists()


def test_migrate_is_idempotent(layout):
    first = CliRunner().invoke(m_command, [])
    second = CliRunner().invoke(m_command, [])

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert "migrated 0 file(s)" in second.output


def test_second_pass_reclaims_already_migrated_files(layout):
    """Copy first, delete later: the documented two-step flow must free space."""
    CliRunner().invoke(m_command, [])
    assert (layout["legacy"] / "u2net.onnx").exists()

    result = CliRunner().invoke(m_command, ["--delete-source"])

    assert result.exit_code == 0
    assert not (layout["legacy"] / "u2net.onnx").exists()
    assert (layout["new"] / "models" / "u2net" / "u2net.onnx").exists()


def test_mismatched_copy_is_left_alone(layout):
    """A truncated destination must not license deleting the source."""
    CliRunner().invoke(m_command, [])

    dst = layout["new"] / "models" / "u2net" / "u2net.onnx"
    dst.write_text("")  # simulate a partial copy

    CliRunner().invoke(m_command, ["--delete-source"])

    assert (layout["legacy"] / "u2net.onnx").exists()


def test_missing_legacy_dir_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.delenv("U2NET_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "nothing_here"))
    monkeypatch.setenv("REMBG_HOME", str(tmp_path / "new"))

    result = CliRunner().invoke(m_command, [])

    assert result.exit_code == 0
    assert "nothing to migrate" in result.output


def test_same_source_and_target_is_refused(monkeypatch, tmp_path):
    """With U2NET_HOME set, both roots collapse to one; migrating is a no-op."""
    monkeypatch.setenv("U2NET_HOME", str(tmp_path / "shared"))
    monkeypatch.delenv("REMBG_HOME", raising=False)
    (tmp_path / "shared").mkdir()

    result = CliRunner().invoke(m_command, [])

    assert result.exit_code == 0
    assert "nothing to migrate" in result.output
