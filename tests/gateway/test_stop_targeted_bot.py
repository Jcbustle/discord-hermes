"""Per-bot stop: ``/stop @bot`` interrupts only that bot's run in the same chat.

Issue #123928 layer A: in a multi-bot group chat a bare ``/stop`` deliberately
stops EVERY run in the room (``_chat_scoped_run_keys``). ``/stop @botname``
must stop only the named bot's run and leave other bots' in-flight work alone;
targeting a bot with nothing running must stop nothing.
"""

import pytest

from agent.i18n import t
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.platforms.base import Platform
from gateway.platforms.event import MessageEvent, MessageType


class _FakeAgent:
    pass


class _StoreEntry:
    def __init__(self, session_key):
        self.session_key = session_key


class _FakeStore:
    def __init__(self, session_key):
        self._key = session_key

    def get_or_create_session(self, source):
        return _StoreEntry(self._key)


def _slack_group(user_id, profile=None):
    source = SessionSource(
        platform=Platform.SLACK, chat_type="group", chat_id="C9",
        user_id=user_id, scope_id="T1",
    )
    key = build_session_key(source, profile=profile) if profile else build_session_key(source)
    return source, key


def _runner_with_runs(running_keys, own_key, authorized=True):
    runner = object.__new__(GatewayRunner)
    runner._running_agents = dict.fromkeys(running_keys, _FakeAgent())
    runner.session_store = _FakeStore(own_key)
    runner._is_user_authorized_for_source = lambda source, **kw: authorized
    runner.adapters = {}
    interrupted = []

    async def _fake_interrupt(session_key, source, *, interrupt_reason, invalidation_reason,
                              release_running_state=True):
        interrupted.append((session_key, invalidation_reason))

    runner._interrupt_and_clear_session = _fake_interrupt
    return runner, interrupted


async def _stop(text, stop_source, running_keys, authorized=True):
    own_key = build_session_key(stop_source)
    runner, interrupted = _runner_with_runs(running_keys, own_key, authorized=authorized)
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=stop_source)
    result = await runner._handle_stop_command(event)
    return interrupted, result


@pytest.mark.asyncio
async def test_stop_at_bot_only_stops_that_bot():
    _, main_key = _slack_group("U-bob")
    _, work_key = _slack_group("U-bob", profile="work")
    assert main_key != work_key
    stop_source, _ = _slack_group("U-alice")

    interrupted, result = await _stop("/stop @work", stop_source, [main_key, work_key])

    assert interrupted == [(work_key, "stop_command_targeted")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_at_idle_bot_stops_nothing():
    _, main_key = _slack_group("U-bob")
    _, work_key = _slack_group("U-bob", profile="work")
    stop_source, _ = _slack_group("U-alice")

    interrupted, result = await _stop("/stop @idle", stop_source, [main_key, work_key])

    assert interrupted == []
    assert result == t("gateway.stop.no_active")


@pytest.mark.asyncio
async def test_bare_stop_keeps_own_profile_scope():
    # Untargeted /stop keeps its historical scope: the caller's own profile namespace only
    # (cross-profile isolation is the established contract — see
    # test_stop_thread_sibling.py::test_sibling_does_not_cross_profiles). Reaching another bot
    # takes an explicit, authorized `/stop @bot`.
    _, main_key = _slack_group("U-bob")
    _, work_key = _slack_group("U-bob", profile="work")
    stop_source, _ = _slack_group("U-alice")

    interrupted, result = await _stop("/stop", stop_source, [main_key, work_key])

    assert [k for k, _ in interrupted] == [main_key]
    assert result == t("gateway.stop.stopped")


# ---------------------------------------------------------------------------
# Review follow-ups (#123928): one shared target resolver for all /stop entries
# ---------------------------------------------------------------------------


def _event(text, source):
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


@pytest.mark.asyncio
async def test_busy_stop_at_other_bot_delegates_and_spares_the_caller():
    """P1 (review): while the caller's own session is running, ``/stop @work`` used to
    take the busy hard-kill path and interrupt the CALLER. The target must go through
    the shared resolver and stop only the named bot."""
    stop_source, own_key = _slack_group("U-alice")
    _, main_key = _slack_group("U-bob")
    _, work_key = _slack_group("U-bob", profile="work")
    runner, interrupted = _runner_with_runs([own_key, main_key, work_key], own_key)

    result = await runner._busy_stop_command(
        _event("/stop @work", stop_source), own_key, stop_source)

    assert interrupted == [(work_key, "stop_command_targeted")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_busy_stop_own_target_keeps_hard_kill_fast_path():
    stop_source, own_key = _slack_group("U-alice")
    runner, interrupted = _runner_with_runs([own_key], own_key)

    result = await runner._busy_stop_command(
        _event("/stop", stop_source), own_key, stop_source)

    assert interrupted == [(own_key, "stop_command")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_at_main_reaches_the_default_bot():
    """P2 (review): ``main`` is the default bot's spelling to users (its keys live under
    ``agent:main``); ``/stop @main`` used to resolve to the marked ``agent:main~``
    namespace only and silently stop nothing."""
    stop_source, _ = _slack_group("U-alice")
    _, main_key = _slack_group("U-bob")
    _, work_key = _slack_group("U-bob", profile="work")

    interrupted, result = await _stop("/stop @main", stop_source, [main_key, work_key])

    assert interrupted == [(main_key, "stop_command_targeted")]
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_at_main_also_covers_the_marked_main_profile():
    """The marked ``agent:main~`` namespace (a profile literally named ``main``) has no
    other user-facing spelling: ``@main`` covers it and the default alike."""
    stop_source, _ = _slack_group("U-alice")
    _, main_key = _slack_group("U-bob")
    _, marked_key = _slack_group("U-bob", profile="main")
    assert marked_key.startswith("agent:main~:")

    interrupted, result = await _stop("/stop @main", stop_source, [main_key, marked_key])

    assert {key for key, _ in interrupted} == {main_key, marked_key}
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_stop_at_default_still_names_the_default_bot():
    stop_source, _ = _slack_group("U-alice")
    _, main_key = _slack_group("U-bob")
    _, work_key = _slack_group("U-bob", profile="work")

    interrupted, result = await _stop("/stop @default", stop_source, [main_key, work_key])

    assert interrupted == [(main_key, "stop_command_targeted")]


@pytest.mark.asyncio
async def test_pending_sentinel_stop_at_other_delegates_instead_of_unlocking():
    """Review :693: a ``/stop @other`` arriving while the CALLER's session is only
    starting (pending sentinel) used to force-clear the caller's sentinel; the target
    must be delegated to the scoped handler instead."""
    from types import SimpleNamespace

    from gateway.run import _AGENT_PENDING_SENTINEL

    stop_source, own_key = _slack_group("U-alice")
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    released = []
    runner._release_running_agent_state = lambda key: released.append(key)

    async def _fake_busy_slash(event, source, quick_key):
        return False, None

    runner._hm_busy_slash_or_photo = _fake_busy_slash
    runner._effective_busy_input_mode = lambda source: "interrupt"
    runner._peek_session_state = lambda key: SimpleNamespace(
        turn=SimpleNamespace(agent=_AGENT_PENDING_SENTINEL, started_ts=0))
    delegated = []

    async def _fake_stop_handler(event):
        delegated.append(event.get_command_args())
        return t("gateway.stop.stopped")

    runner._handle_stop_command = _fake_stop_handler

    result = await runner._hm_handle_running_session_message(
        _event("/stop @work", stop_source), stop_source, own_key)

    assert delegated == ["@work"]
    assert released == []
    assert result == t("gateway.stop.stopped")


@pytest.mark.asyncio
async def test_pending_sentinel_bare_stop_still_unlocks_the_session():
    from types import SimpleNamespace

    from gateway.run import _AGENT_PENDING_SENTINEL

    stop_source, own_key = _slack_group("U-alice")
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    released = []
    runner._release_running_agent_state = lambda key: released.append(key)

    async def _fake_busy_slash(event, source, quick_key):
        return False, None

    runner._hm_busy_slash_or_photo = _fake_busy_slash
    runner._effective_busy_input_mode = lambda source: "interrupt"
    runner._peek_session_state = lambda key: SimpleNamespace(
        turn=SimpleNamespace(agent=_AGENT_PENDING_SENTINEL, started_ts=0))

    result = await runner._hm_handle_running_session_message(
        _event("/stop", stop_source), stop_source, own_key)

    assert released == [own_key]
    assert result is not None
