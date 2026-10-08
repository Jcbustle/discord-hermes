"""RED test for #105293: block unmanaged input synthesis + driver kill via terminal.

Incident: agent escaped computer_use into terminal, ran raw Win32
keybd_event without try/finally (Alt stuck down, host needed reboot),
then taskkill /F cua-driver.exe severed the MCP transport with no
circuit breaker. The managed driver already auto-revives ended sessions
and fails closed on transport drops; the open hole is the terminal
escape hatch. One guard in the shared pre-exec chain closes it.
"""

import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from tools.terminal_tool_guards import unmanaged_input_block


HOSTILE = [
    'python -c "import win32api, win32con; win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)"',
    "python -c \"from ctypes import windll; windll.user32.SendInput(1, 0, 0)\"",
    "taskkill /F /IM cua-driver.exe",
    "taskkill /f /im cua-driver.exe",
    "powershell -c \"Stop-Process -Name cua-driver -Force\"",
    # Name-before-verb soup: the same kill is spelled object-first once piped.
    "powershell -c \"Get-Process cua-driver | Stop-Process -Force\"",
    "wmic process where name='cua-driver.exe' call terminate",
]

BENIGN = [
    "echo hello",
    "python -m pytest tests/tools/test_terminal_self_repo_guard.py -q",
    "taskkill /F /IM notepad.exe",
    "git status --short",
    "Get-Process cua-driver",  # listing the driver is not killing it
]


@pytest.mark.parametrize("cmd", HOSTILE)
def test_blocks_unmanaged_input_and_driver_kill(cmd):
    blocked = unmanaged_input_block(command=cmd)
    assert blocked is not None, f"not blocked: {cmd}"
    payload = json.loads(blocked)
    assert payload["exit_code"] == 1
    assert payload["status"] == "blocked"


@pytest.mark.parametrize("cmd", BENIGN)
def test_benign_passes(cmd):
    assert unmanaged_input_block(command=cmd) is None


def _make_env_config(**overrides):
    config = {
        "env_type": "local",
        "timeout": 180,
        "cwd": "/tmp",
        "host_cwd": None,
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
    }
    config.update(overrides)
    return config


def _run(command, config, **kwargs):
    from tools.terminal_tool import terminal_tool

    mock_env = MagicMock()
    mock_env.execute.return_value = {"output": "ok", "returncode": 0}
    mock_env.cwd = config["cwd"]
    with ExitStack() as stack:
        stack.enter_context(
            patch("tools.terminal_tool._get_env_config", return_value=config)
        )
        stack.enter_context(patch("tools.terminal_tool._start_cleanup_thread"))
        stack.enter_context(
            patch("tools.terminal_tool._active_environments", {"default": mock_env})
        )
        stack.enter_context(patch("tools.terminal_tool._last_activity", {"default": 0}))
        stack.enter_context(patch("tools.terminal_tool._session_cwd", {}))
        stack.enter_context(
            patch(
                "tools.terminal_tool._check_all_guards", return_value={"approved": True}
            )
        )
        result = json.loads(terminal_tool(command=command, **kwargs))
    return result, mock_env


def test_wiring_blocks_keybd_event_before_execute():
    config = _make_env_config()
    result, env = _run(
        'python -c "import win32api; win32api.keybd_event(18, 0, 0, 0)"', config
    )
    assert result["status"] == "blocked"
    env.execute.assert_not_called()


def test_wiring_blocks_driver_kill_even_with_force():
    config = _make_env_config()
    result, env = _run("taskkill /F /IM cua-driver.exe", config, force=True)
    assert result["status"] == "blocked"
    env.execute.assert_not_called()


def _run_execute_code(monkeypatch, code):
    """Drive the real execute_code entry with the post-guard path stubbed, so a script
    that slips past the guard is recorded instead of executed."""
    import tools.code_execution_tool as cet
    import tools.process_registry as process_registry
    import tools.approval as approval_module

    monkeypatch.setattr(cet, "SANDBOX_AVAILABLE", True)
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
    monkeypatch.setattr(cet, "_get_env_config", lambda: {"env_type": "local"}, raising=False)
    monkeypatch.setattr(
        approval_module, "check_execute_code_guard", lambda *a, **k: {"approved": True}, raising=False
    )
    executed: list[str] = []
    monkeypatch.setattr(
        "tools.code_kernel.execute_in_session_kernel",
        lambda code, **kw: executed.append(code) or json.dumps({"status": "ok"}),
        raising=False,
    )
    result = json.loads(cet.execute_code(code))
    return result, executed


def test_execute_code_blocks_unmanaged_input(monkeypatch):
    """#105293 sibling surface: the terminal pre-exec refusal must cover execute_code,
    or `os.system("... keybd_event ...")` is a straight bypass (the lifecycle-guard
    shape that is already mirrored there)."""
    result, executed = _run_execute_code(
        monkeypatch,
        "import os; os.system('python -c \"import win32api; win32api.keybd_event(18, 0, 0, 0)\"')",
    )
    assert "raw input synthesis" in result.get("error", ""), result
    assert executed == [], "hostile script reached the kernel"


def test_execute_code_blocks_driver_kill(monkeypatch):
    result, executed = _run_execute_code(
        monkeypatch, "import os; os.system('taskkill /F /IM cua-driver.exe')"
    )
    assert "cua-driver" in result.get("error", ""), result
    assert executed == [], "driver kill reached the kernel"
