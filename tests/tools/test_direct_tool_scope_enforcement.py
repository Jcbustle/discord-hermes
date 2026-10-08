"""Direct tool calls must honor the session toolset scope (#121089).

``platform_toolsets`` / ``disabled_toolsets`` hid restricted tools from the
advertised definitions, but ``handle_function_call`` only scoped the Tool
Search bridge (``tool_search`` / ``tool_call``). A direct call such as
``terminal(...)`` executed regardless of scope, so a platform configured
without the terminal toolset (e.g. Telegram with ``[file, skills, vision]``)
could still run shell commands.
"""

import json

import model_tools
from model_tools import handle_function_call

RESTRICTED_ENABLED = ["file", "skills", "vision"]
RESTRICTED_DISABLED = ["terminal"]

_SCOPE_KWARGS = dict(
    skip_pre_tool_call_hook=True,
    skip_tool_request_middleware=True,
    skip_tool_execution_middleware=True,
)


def _patch_dispatch(monkeypatch):
    calls = []

    def fake_dispatch(name, args, **kwargs):
        calls.append(name)
        return '{"ok": true}'

    monkeypatch.setattr(model_tools.registry, "dispatch", fake_dispatch)
    return calls


class TestDirectCallScopeEnforcement:
    def test_terminal_tool_is_registered(self):
        """Guard against a vacuous pass: the refusal must be the scope check."""
        names = model_tools._select_tool_names(None, None, quiet_mode=True)
        assert "terminal" in names

    def test_out_of_scope_direct_call_is_refused_without_executing(self, monkeypatch):
        calls = _patch_dispatch(monkeypatch)
        result = json.loads(handle_function_call(
            "terminal", {"command": "uptime"},
            enabled_toolsets=list(RESTRICTED_ENABLED),
            disabled_toolsets=list(RESTRICTED_DISABLED),
            **_SCOPE_KWARGS,
        ))
        assert calls == [], "out-of-scope direct call reached the registry"
        assert "not available in this session" in result.get("error", "")

    def test_disabled_toolset_is_subtracted_from_full_grant(self, monkeypatch):
        """No enabled allowlist: disabled terminal alone must still block."""
        calls = _patch_dispatch(monkeypatch)
        result = json.loads(handle_function_call(
            "terminal", {"command": "uptime"},
            disabled_toolsets=list(RESTRICTED_DISABLED),
            **_SCOPE_KWARGS,
        ))
        assert calls == []
        assert "not available in this session" in result.get("error", "")

    def test_in_scope_direct_call_still_executes(self, monkeypatch):
        calls = _patch_dispatch(monkeypatch)
        result = json.loads(handle_function_call(
            "terminal", {"command": "uptime"},
            enabled_toolsets=["terminal"],
            **_SCOPE_KWARGS,
        ))
        assert calls == ["terminal"]
        assert result.get("ok") is True

    def test_unrestricted_session_preserves_direct_execution(self, monkeypatch):
        calls = _patch_dispatch(monkeypatch)
        result = json.loads(handle_function_call(
            "terminal", {"command": "uptime"},
            **_SCOPE_KWARGS,
        ))
        assert calls == ["terminal"]
        assert result.get("ok") is True


class TestExecuteCodeSandboxScope:
    """The execute_code sandbox must inherit the session's scope (#121089 review).

    The sandbox dispatches host tools through its own RPC allow-list; if the
    allow-list falls back to the full set when the session's grant shares
    nothing with it, a session configured without ``terminal`` can still run
    shell commands from inside ``execute_code``.
    """

    def test_empty_intersection_fails_closed(self):
        from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, _sandbox_tools_for

        assert _sandbox_tools_for(["execute_code"]) == frozenset()
        assert _sandbox_tools_for([]) == frozenset()
        assert _sandbox_tools_for(["terminal"]) == frozenset({"terminal"})
        # No grant list at all keeps the unrestricted legacy behavior.
        assert _sandbox_tools_for(None) == frozenset(SANDBOX_ALLOWED_TOOLS)

    def test_sandbox_rpc_refuses_terminal_when_grant_has_no_overlap(self):
        import time as _time

        from tools.code_execution_rpc import _handle_rpc_request
        from tools.code_execution_tool import _sandbox_tools_for

        allowed = _sandbox_tools_for(["execute_code"])
        dispatched = []
        response = _handle_rpc_request(
            {"tool": "terminal", "args": {"command": "echo pwn"}},
            allowed_tools=allowed, tool_call_counter=[0], max_tool_calls=5,
            dispatch=lambda name, args: dispatched.append(name),
            tool_call_log=[], call_start=_time.monotonic(), where="unit",
        )
        assert dispatched == [], "sandbox RPC dispatched a tool outside the session grant"
        assert "not available in execute_code" in response
