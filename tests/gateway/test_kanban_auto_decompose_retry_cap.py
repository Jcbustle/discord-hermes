"""Attempt cap for auto-decompose on a failing triage card (issue #118603).

Without a cap, ``auto_decompose_tick`` retries a card whose decomposition
fails on *every* dispatcher tick, forever — burning dispatcher budget and
(reported) re-billing one aux LLM call per tick with no cost attribution.
Connection-class failures are exempt from the cap: the call never reached the
model, so they are retried next tick instead of spending the card's budget
(#118603 review).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from gateway import kanban_watchers_dispatcher as kwd


def _dispatcher():
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def test_failing_triage_card_is_not_retried_every_tick_forever(monkeypatch):
    """A card that fails 8 straight ticks must be billed at most 3 times."""
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    bills: list[str] = []

    def fake_decompose(task_id, author=None):
        bills.append(task_id)
        return SimpleNamespace(ok=False, fanout=False, child_ids=None,
                               reason="LLM error: InternalServerError")

    fake = SimpleNamespace(list_triage_ids=lambda: ["t_doomed"], decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)

    dispatcher = _dispatcher()
    for _ in range(8):
        dispatcher.auto_decompose_tick(10)

    assert len(bills) <= 3, (
        f"doomed card billed {len(bills)} times over 8 dispatcher ticks — "
        "no attempt cap (issue #118603)"
    )


def test_parked_card_is_retried_after_the_retry_window(monkeypatch):
    """A counted failure (unusable reply) parks the card, but the retry window
    grants a fresh budget so it does not stay parked until restart (#118603)."""
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    now = [1_000_000.0]
    monkeypatch.setattr(kwd.time, "monotonic", lambda: now[0])

    upstream_ok = {"v": False}

    def fake_decompose(task_id, author=None):
        if upstream_ok["v"]:
            return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)
        return SimpleNamespace(ok=False, fanout=False, child_ids=None,
                               reason="LLM returned malformed JSON")

    fake = SimpleNamespace(list_triage_ids=lambda: ["t_parked"], decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)

    dispatcher = _dispatcher()
    for _ in range(3):
        dispatcher.auto_decompose_tick(10)
    assert dispatcher._decompose_failures[("default", "t_parked")][0] == 3

    # Upstream recovers, but the card stays parked inside the window.
    upstream_ok["v"] = True
    for _ in range(5):
        assert dispatcher.auto_decompose_tick(10) == 0

    # Once the window elapses the card is retried and decomposes.
    now[0] += dispatcher._DECOMPOSE_RETRY_WINDOW_SECONDS + 1
    assert dispatcher.auto_decompose_tick(10) == 1


def test_transient_connection_error_does_not_consume_the_budget(monkeypatch):
    """A connection-class failure (the call never reached the model) must not
    spend the card's attempt budget — once upstream is reachable again the very
    next tick decomposes it, with no window wait (#118603 review)."""
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])

    upstream_ok = {"v": False}

    def fake_decompose(task_id, author=None):
        if upstream_ok["v"]:
            return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)
        return SimpleNamespace(ok=False, fanout=False, child_ids=None,
                               reason="LLM error: APIConnectionError")

    fake = SimpleNamespace(list_triage_ids=lambda: ["t_blip"], decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)

    dispatcher = _dispatcher()
    for _ in range(3):
        assert dispatcher.auto_decompose_tick(10) == 0
    assert ("default", "t_blip") not in dispatcher._decompose_failures, (
        "a connection blip spent the card's attempt budget"
    )

    upstream_ok["v"] = True
    assert dispatcher.auto_decompose_tick(10) == 1


def test_transient_blip_neither_consumes_nor_resets_the_budget(monkeypatch):
    """A blip in the middle of counted failures is free — and must not reset
    the streak: a card that really cannot decompose still caps at 3 bills."""
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    reasons = [
        "LLM returned malformed JSON",
        "LLM returned malformed JSON",
        "LLM error: APIConnectionError",
        "LLM returned malformed JSON",
        "LLM returned malformed JSON",
    ]
    bills: list[str] = []

    def fake_decompose(task_id, author=None):
        bills.append(task_id)
        return SimpleNamespace(ok=False, fanout=False, child_ids=None,
                               reason=reasons[min(len(bills), len(reasons)) - 1])

    fake = SimpleNamespace(list_triage_ids=lambda: ["t_mixed"], decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)

    dispatcher = _dispatcher()
    for _ in range(8):
        dispatcher.auto_decompose_tick(10)

    assert len(bills) == 4, (
        f"mixed-blip card billed {len(bills)} times; the blip must be free and "
        "the three counted failures must park the card"
    )
    assert dispatcher._decompose_failures[("default", "t_mixed")][0] == 3


def test_list_read_failure_keeps_the_failure_budgets(monkeypatch):
    """A transient list_triage_ids() failure must not read as "all cards left
    triage" and wipe the whole board's budgets (#118603 review)."""
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])
    list_fails = {"v": False}

    def fake_list():
        if list_fails["v"]:
            raise RuntimeError("database is locked")
        return ["t_card"]

    def fake_decompose(task_id, author=None):
        return SimpleNamespace(ok=False, fanout=False, child_ids=None, reason="boom")

    fake = SimpleNamespace(list_triage_ids=fake_list, decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)

    dispatcher = _dispatcher()
    for _ in range(2):
        dispatcher.auto_decompose_tick(10)
    assert dispatcher._decompose_failures[("default", "t_card")][0] == 2

    list_fails["v"] = True
    dispatcher.auto_decompose_tick(10)  # listing raises this tick
    assert dispatcher._decompose_failures[("default", "t_card")][0] == 2, (
        "a failed list read reset the board's decompose budgets"
    )
