import os

import pytest

from rembg.sessions.base import BaseSession
from rembg.sessions.u2net import U2netSession


@pytest.fixture
def clean_env(monkeypatch):
    """Start from an environment where none of the path variables are set."""
    for var in ("U2NET_HOME", "REMBG_HOME", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_defaults_to_dot_rembg(clean_env):
    assert U2netSession.rembg_home() == os.path.expanduser("~/.rembg")


def test_rembg_home_env_wins_over_default(clean_env):
    clean_env.setenv("REMBG_HOME", "/tmp/somewhere")
    assert U2netSession.rembg_home() == "/tmp/somewhere"


def test_xdg_data_home_has_no_leading_dot(clean_env):
    """XDG_DATA_HOME is already a data directory, so the child is not hidden."""
    clean_env.setenv("XDG_DATA_HOME", "/tmp/xdg")
    assert U2netSession.rembg_home() == "/tmp/xdg/rembg"


def test_legacy_env_still_takes_precedence(clean_env):
    """Anyone already pointing U2NET_HOME somewhere keeps that directory."""
    clean_env.setenv("U2NET_HOME", "/tmp/legacy")
    clean_env.setenv("REMBG_HOME", "/tmp/new")
    assert U2netSession.rembg_home() == "/tmp/legacy"


def test_u2net_home_is_an_alias_for_rembg_home(clean_env):
    assert U2netSession.u2net_home() == U2netSession.rembg_home()


def test_model_dir_is_per_model(clean_env):
    clean_env.setenv("REMBG_HOME", "/tmp/home")
    assert U2netSession.model_dir() == "/tmp/home/models/u2net"


def test_resolve_existing_prefers_new_layout(clean_env, tmp_path):
    clean_env.setenv("REMBG_HOME", str(tmp_path / "new"))
    clean_env.setenv("U2NET_HOME", "")  # empty means unset for the lookup below

    new_dir = tmp_path / "new" / "models" / "u2net"
    new_dir.mkdir(parents=True)
    (new_dir / "u2net.onnx").write_text("new")

    assert U2netSession.resolve_existing("u2net.onnx") == str(new_dir / "u2net.onnx")


def test_resolve_existing_falls_back_to_legacy(clean_env, tmp_path):
    """A model downloaded by an older version must not be fetched again."""
    clean_env.setenv("REMBG_HOME", str(tmp_path / "new"))
    clean_env.setenv("XDG_DATA_HOME", str(tmp_path / "legacy_base"))

    legacy_dir = tmp_path / "legacy_base" / ".u2net"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "u2net.onnx").write_text("old")

    assert U2netSession.resolve_existing("u2net.onnx") == str(legacy_dir / "u2net.onnx")


def test_resolve_existing_returns_none_when_absent(clean_env, tmp_path):
    clean_env.setenv("REMBG_HOME", str(tmp_path / "new"))
    clean_env.setenv("XDG_DATA_HOME", str(tmp_path / "legacy_base"))

    assert U2netSession.resolve_existing("u2net.onnx") is None


def test_validate_model_path_accepts_both_roots(clean_env, tmp_path):
    clean_env.setenv("REMBG_HOME", str(tmp_path / "new"))
    clean_env.setenv("XDG_DATA_HOME", str(tmp_path / "legacy_base"))

    inside_new = str(tmp_path / "new" / "models" / "custom" / "m.onnx")
    inside_legacy = str(tmp_path / "legacy_base" / ".u2net" / "m.onnx")

    assert BaseSession.validate_model_path(model_path=inside_new) == inside_new
    assert BaseSession.validate_model_path(model_path=inside_legacy) == inside_legacy


def test_validate_model_path_rejects_outside_paths(clean_env, tmp_path):
    clean_env.setenv("REMBG_HOME", str(tmp_path / "new"))

    with pytest.raises(ValueError):
        BaseSession.validate_model_path(model_path="/etc/passwd")


def test_validate_model_path_rejects_traversal(clean_env, tmp_path):
    clean_env.setenv("REMBG_HOME", str(tmp_path / "new"))

    with pytest.raises(ValueError):
        BaseSession.validate_model_path(
            model_path=str(tmp_path / "new" / ".." / ".." / "etc" / "passwd")
        )


def test_validate_model_path_rejects_sibling_prefix(clean_env, tmp_path):
    """`/x/.rembg-evil` must not pass because it starts with `/x/.rembg`."""
    clean_env.setenv("REMBG_HOME", str(tmp_path / "new"))

    with pytest.raises(ValueError):
        BaseSession.validate_model_path(
            model_path=str(tmp_path / "new-evil" / "m.onnx")
        )


def test_validate_model_path_requires_the_argument(clean_env):
    with pytest.raises(ValueError):
        BaseSession.validate_model_path()
