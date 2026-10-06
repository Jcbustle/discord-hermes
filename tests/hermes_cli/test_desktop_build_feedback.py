"""The desktop pre-launch build must announce itself before it blocks.

The XDG entry launches ``hermes desktop`` with ``Terminal=false``, so the
multi-minute pre-launch build is indistinguishable from a hang unless the
launcher writes the wait to stderr *before* the long child starts (#127619).

Regression for #127619.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import main_desktop


def _desktop(tmp_path: Path) -> Path:
    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    return desktop


def _capture_phases(monkeypatch, capsys) -> list[tuple[list[str], str]]:
    """Record each long build phase with what stderr held when it started."""
    import pm
    import pm.progress

    seen: list[tuple[list[str], str]] = []

    def run(command, label, **_kwargs):
        seen.append((list(command), capsys.readouterr().err))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pm.progress, "run_contained", run)
    # Packaged builds stage PM's pinned git (Windows only); the test tree has no
    # PM environment, so hand back the caller's env untouched.
    monkeypatch.setattr(pm, "ensure", lambda *_names, base_env: SimpleNamespace(env=base_env))
    return seen


def test_source_build_announces_the_wait_on_stderr_before_npm_starts(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    seen = _capture_phases(monkeypatch, capsys)

    main_desktop.build_prepared_desktop(_desktop(tmp_path), source_mode=True, npm="npm", env={})

    assert [command[2] for command, _ in seen] == ["build"]
    notice = seen[0][1]
    assert notice, "npm started with nothing on stderr to explain the wait"
    assert "few minutes" in notice


def test_packaged_build_announces_both_long_phases_on_stderr(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    desktop = _desktop(tmp_path)
    seen = _capture_phases(monkeypatch, capsys)
    # The host's own build shape: no running Desktop owns this temp tree.
    monkeypatch.setattr(main_desktop, "_desktop_ancestor_in", lambda *_args: None)
    monkeypatch.setattr(main_desktop, "_stop_desktop_processes_locking_build", lambda *_args: [])
    monkeypatch.setattr(main_desktop, "_promote_staged_desktop_app", lambda *_args, **_kw: desktop / "Hermes")

    main_desktop.build_prepared_desktop(desktop, source_mode=False, npm="npm", env={"PATH": ""})

    assert [command[2] for command, _ in seen] == ["build", "builder"]
    for command, notice in seen:
        assert "few minutes" in notice, f"{command[2]} started with nothing on stderr: {notice!r}"


def test_desktop_command_announces_the_dependency_install_before_it_starts(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """The dependency install is the first long phase ``hermes desktop`` runs.

    ``cmd_gui`` reaches ``prepare_source_dependencies`` before
    ``build_prepared_desktop``, so without a notice there the first thing a
    waiting user sees is still nothing at all. Snapshot stderr exactly as the
    child would be handed control.
    """
    from hermes_cli import main as cli_main
    from hermes_cli import source_build

    root = tmp_path / "hermes-agent"
    desktop = root / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli_main, "PROJECT_ROOT", root)
    monkeypatch.setattr(main_desktop, "_desktop_launch_env", lambda args: ({}, []))
    monkeypatch.setattr(main_desktop, "_desktop_build_needed", lambda *args, **kwargs: True)
    monkeypatch.setattr(source_build, "source_build_env", lambda env=None, **kwargs: {"PATH": ""})
    monkeypatch.setattr(main_desktop.shutil, "which", lambda *args, **kwargs: "npm")

    seen: dict[str, str] = {}

    class _Stop(Exception):
        pass

    def prepare(*args, **kwargs):
        seen["stderr"] = capsys.readouterr().err
        raise _Stop

    monkeypatch.setattr(source_build, "prepare_source_dependencies", prepare)

    args = argparse.Namespace(
        skip_build=False, build_only=False, force_build=False, source=False,
        fake_boot=False, ignore_existing=False, hermes_root=None, cwd=None,
        setup_tcc_identity=False, identity=None,
    )
    with pytest.raises(_Stop):
        main_desktop.cmd_gui(args)

    assert "Installing Node dependencies" in seen["stderr"]
    assert "few minutes" in seen["stderr"]
