"""Regression tests for #119143: typing must stay invisible during inbound
enrichment (STT/vision) and appear only once the agent run starts.

Contract:
- A gateway-managed adapter (``gateway_runner`` set — production wiring in
  ``run_adapters._create_adapter``) holds the refresh silent from spawn until the
  runner signals readiness via ``resume_typing_for_chat()`` immediately before
  ``_run_agent``. Early returns that never start an agent turn never signal, so they
  produce zero typing.
- A bare adapter (no runner) keeps the unconditional refresh contract, so a handler
  that never signals is not starved of typing (see test_typing_indicator_toggle).
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _Adapter(BasePlatformAdapter):
    def __init__(self, *, managed: bool = False):
        super().__init__(PlatformConfig(enabled=True, token="fake"), Platform.TELEGRAM)
        self._busy_text_mode = ""
        self.typing_ts = []
        self.resume_ts = []
        if managed:
            # Mirror run_adapters._create_adapter: production adapters always see their runner.
            self.gateway_runner = object()

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        self.typing_ts.append(time.monotonic())
        return None

    async def stop_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}

    def resume_typing_for_chat(self, chat_id):
        self.resume_ts.append(time.monotonic())
        super().resume_typing_for_chat(chat_id)


def _event(chat_id="c1"):
    return MessageEvent(
        text="voice note",
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="private"
        ),
        message_id="m1",
    )


@pytest.mark.asyncio
async def test_bare_adapter_shows_typing_without_any_readiness_signal():
    """No runner attached: the refresh fires purely on ``typing_indicator``, so a
    handler that never signals readiness is not starved (the existing toggle test's
    contract, made explicit)."""
    adapter = _Adapter(managed=False)

    async def handler(_event):
        await asyncio.sleep(0.4)
        return "done"

    adapter.set_message_handler(handler)
    event = _event()
    await adapter._process_message_background(event, build_session_key(event.source))
    assert adapter.typing_ts, "bare adapter must refresh typing without a readiness signal"


@pytest.mark.asyncio
async def test_managed_adapter_gates_typing_until_readiness_signal():
    """Gateway-managed adapter: zero ``send_typing`` during the STT window, at least
    one after the runner signals readiness."""
    adapter = _Adapter(managed=True)
    marks = {}

    async def handler(event):
        marks["pre_start"] = time.monotonic()
        await asyncio.sleep(2.5)  # inbound STT / transcription window
        marks["pre_end"] = time.monotonic()
        # runner protocol: readiness immediately before the agent run
        adapter.resume_typing_for_chat(event.source.chat_id)
        marks["ready"] = time.monotonic()
        await asyncio.sleep(4.5)  # agent run window (>2 typing intervals)
        return "done"

    adapter.set_message_handler(handler)
    event = _event()
    await adapter._process_message_background(event, build_session_key(event.source))

    pre = [t for t in adapter.typing_ts if t < marks["ready"]]
    post = [t for t in adapter.typing_ts if t >= marks["ready"]]
    assert marks["pre_end"] - marks["pre_start"] >= 2.0, "STT window too short"
    assert pre == [], f"typing fired during inbound enrichment: {len(pre)} calls"
    assert len(post) >= 1, "no typing after agent run started"


def _prepared_turn(runner):
    return runner._PreparedTurn([], "", "hello", "hello", None, None)


@pytest.mark.asyncio
async def test_runner_resumes_typing_before_run_agent_on_the_real_path():
    """Drives the real ``_handle_message_with_agent``: readiness must be signalled
    before ``_run_agent`` (replacing the resume block with ``pass`` makes this fail)
    and no ``send_typing`` may fire before it."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._run_in_executor_with_context = asyncio.to_thread
    adapter = _Adapter(managed=True)
    runner._delivery_adapter_for = lambda source: adapter
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c-119143", chat_type="private"
    )
    key = build_session_key(source)
    event = _event(chat_id="c-119143")
    windows = {}
    order = {}

    async def resolve(event_, source_):
        return (source, SimpleNamespace(session_id="sess-119143"), key)

    async def prepare(*_args, **_kwargs):
        windows["prep_start"] = time.monotonic()
        await asyncio.sleep(2.5)  # STT/vision enrichment window
        windows["prep_end"] = time.monotonic()
        return _prepared_turn(runner), {}

    async def model(**_kwargs):
        order["run_agent"] = time.monotonic()
        await asyncio.sleep(3.5)  # agent run window (>1 typing interval after resume)
        raise asyncio.CancelledError  # stop before unrelated post-turn delivery

    runner._hmwa_resolve_session = resolve
    runner._hmwa_prepare_turn = prepare
    runner._run_agent = model
    runner.hooks = SimpleNamespace(emit=AsyncMock())
    runner._pinned_channel_inputs = lambda *_a, **_k: ("", source)
    runner._persist_prompt_pins = AsyncMock()
    adapter.set_message_handler(
        lambda ev: runner._handle_message_with_agent(ev, source, key, 1)
    )

    try:
        await adapter._process_message_background(event, key)
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0)  # let cancellation callbacks settle

    assert order.get("run_agent"), "stub agent never ran"
    assert adapter.resume_ts, "runner never signalled typing readiness before _run_agent"
    resume_at = adapter.resume_ts[0]
    assert resume_at <= order["run_agent"] + 1e-6, "resume must happen before _run_agent"
    early = [t for t in adapter.typing_ts if t < windows["prep_end"]]
    assert early == [], f"typing fired during turn preparation: {len(early)} calls"
    assert [t for t in adapter.typing_ts if t >= resume_at], "no typing after readiness"
