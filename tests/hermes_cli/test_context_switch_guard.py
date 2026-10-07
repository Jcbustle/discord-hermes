"""Tests for hermes_cli.context_switch_guard."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli.context_switch_guard import merge_preflight_compression_warning
from hermes_cli.model_switch import ModelSwitchResult


def _result(*, model: str = "small-model") -> ModelSwitchResult:
    return ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider="openrouter",
        provider_changed=False,
        api_key="k",
        base_url="https://example.com/v1",
        api_mode="chat_completions",
        provider_label="openrouter",
        model_info={"context_length": 32_000},
    )


def _compressor(
    monkeypatch,
    *,
    context_length: int = 200_000,
    threshold_tokens_cap: int | None = None,
):
    from agent.context_compressor import ContextCompressor

    monkeypatch.setattr(
        "agent.context_compressor.get_model_context_length",
        lambda *a, **k: context_length,
    )
    return ContextCompressor(
        model="big-model",
        threshold_percent=0.5,
        protect_first_n=3,
        protect_last_n=20,
        quiet_mode=True,
        config_context_length=context_length,
        threshold_tokens_cap=threshold_tokens_cap,
    )




def test_merge_appends_to_existing_warning(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (90_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 32_000,
    )
    cc = _compressor(monkeypatch)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )
    result = _result()
    result.warning_message = "expensive"
    merge_preflight_compression_warning(result, agent=agent)
    assert "expensive" in result.warning_message
    assert "preflight compression" in result.warning_message


def test_cap_lowers_the_switch_warning_threshold_below_the_ratio(monkeypatch):
    """The warning quotes the trigger the compressor will install: on a 1M target the ratio alone says
    500K (no warning at 300K in-flight), the cap says less — the guard must warn with the capped number."""
    cap = 256_000
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (300_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 1_000_000,
    )
    cc = _compressor(
        monkeypatch,
        context_length=200_000,
        threshold_tokens_cap=cap,
    )
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )

    result = _result(model="large-model")
    merge_preflight_compression_warning(result, agent=agent)

    assert "preflight compression" in result.warning_message
    assert f"auto-compress at ~{cap:,}" in result.warning_message


def test_custom_provider_context_avoids_false_shrink_warning(monkeypatch):
    """Classic CLI used to omit custom_providers from the shrink warning.

    Repro: switch onto a custom endpoint with models.<id>.context_length=1M
    while session ~147k. Probe fails → hardcoded catalog match on "qwen"
    (131072) → false "Context window shrinks (... → 131,072)" warning, even
    though /model confirmation and the status bar correctly show 1M.
    """
    custom_provs = [
        {
            "name": "qwen-token-plan",
            "base_url": "https://token-plan.example/compatible-mode/v1",
            "models": {
                "qwen3.9-max-preview": {"context_length": 1_048_576},
            },
        }
    ]
    # Force the probe-down path that hit the "qwen" → 131072 catalog match
    # when custom_providers was not threaded through.
    monkeypatch.setattr(
        "agent.model_metadata._resolve_endpoint_context_length",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "agent.model_metadata._query_ollama_api_show",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (147_053, "measured", True),
    )
    cc = _compressor(monkeypatch, context_length=1_000_000)
    agent = SimpleNamespace(
        model="MiniMax-M3",
        provider="minimax",
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="https://api.minimax.example/v1",
        api_key="",
        _custom_providers=custom_provs,
    )
    result = ModelSwitchResult(
        success=True,
        new_model="qwen3.9-max-preview",
        target_provider="qwen-token-plan",
        provider_changed=True,
        api_key="k",
        base_url="https://token-plan.example/compatible-mode/v1",
        api_mode="chat_completions",
        provider_label="qwen-token-plan",
        model_info=None,
    )

    # Explicit custom_providers — no false shrink warning (1M > 147k*2).
    merge_preflight_compression_warning(
        result,
        agent=agent,
        custom_providers=custom_provs,
    )
    assert not result.warning_message

    # Agent snapshot alone (classic CLI historically forgot to pass the kwarg).
    result2 = ModelSwitchResult(
        success=True,
        new_model="qwen3.9-max-preview",
        target_provider="qwen-token-plan",
        provider_changed=True,
        api_key="k",
        base_url="https://token-plan.example/compatible-mode/v1",
        api_mode="chat_completions",
        provider_label="qwen-token-plan",
        model_info=None,
    )
    merge_preflight_compression_warning(result2, agent=agent)
    assert not result2.warning_message

    # Without any custom_providers source, catalog match still warns (131k).
    agent_no_cp = SimpleNamespace(
        model="MiniMax-M3",
        provider="minimax",
        context_compressor=cc,
        compression_enabled=True,
        conversation_history=[],
        base_url="https://api.minimax.example/v1",
        api_key="",
        _custom_providers=None,
    )
    result3 = ModelSwitchResult(
        success=True,
        new_model="qwen3.9-max-preview",
        target_provider="qwen-token-plan",
        provider_changed=True,
        api_key="k",
        base_url="https://token-plan.example/compatible-mode/v1",
        api_mode="chat_completions",
        provider_label="qwen-token-plan",
        model_info=None,
    )
    merge_preflight_compression_warning(result3, agent=agent_no_cp)
    assert result3.warning_message
    assert "preflight compression" in result3.warning_message
    assert "shrinks" in result3.warning_message
    # Must not honor the unused 1M custom override when no providers were passed.
    assert "1,048,576" not in result3.warning_message


def test_cold_read_note_on_a_large_session_that_will_not_compress(monkeypatch):
    """A switch costs a full re-read even when compression is not coming: the new route answers from
    an empty prefix cache. The summary names that cost instead of leaving the stall unexplained."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (90_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )
    result = _result(model="large-model")
    merge_preflight_compression_warning(result, agent=agent)

    assert "no warm prefix cache" in result.warning_message
    assert "90,000 tokens" in result.warning_message
    # Below the trigger, so nothing about compression belongs in the note.
    assert "preflight compression" not in result.warning_message


def test_no_cold_read_note_on_a_small_session(monkeypatch):
    """A prefill on a small history is instant; the summary stays quiet."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (12_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )
    result = _result(model="large-model")
    merge_preflight_compression_warning(result, agent=agent)

    assert not result.warning_message


def test_compression_warning_is_not_doubled_by_the_cold_read_note(monkeypatch):
    """At or above the trigger the compression warning is the whole answer; the re-read is implied."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (160_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )
    result = _result(model="large-model")
    merge_preflight_compression_warning(result, agent=agent)

    assert "preflight compression" in result.warning_message
    assert "no warm prefix cache" not in result.warning_message


def test_cold_read_note_survives_disabled_compression(monkeypatch):
    """Compression off removes the promise of a compress, not the re-read: the whole history still
    goes to the new route, so the switch summary keeps naming the cost."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    cc.last_prompt_tokens = 180_000
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=False,
        base_url="",
        api_key="",
    )
    result = _result(model="large-model")
    merge_preflight_compression_warning(result, agent=agent)

    assert "no warm prefix cache" in result.warning_message
    assert "180,000 tokens" in result.warning_message
    assert "preflight compression" not in result.warning_message


def test_token_heavy_protected_length_history_is_not_vetoed_as_too_short(monkeypatch):
    """R4: a small message *count* is not a small payload.

    The count that used to veto the estimate and then veto the compression promise
    (``protect_first_n + protect_last_n + 1``) is not what the engine admits on: retention is
    token-bounded, so 24 heavy rows are compressible and the summary says what the switch costs
    instead of falling back to the cold-read note.
    """
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 40_000}
        for i in range(24)
    ]
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )
    result = _result(model="large-model")
    merge_preflight_compression_warning(result, agent=agent, messages=messages)

    assert "past that trigger" in result.warning_message
    assert "preflight compression" in result.warning_message
    assert "no warm prefix cache" not in result.warning_message


def test_estimate_tokens_sizes_a_protected_length_history(monkeypatch):
    """The protected-length early return belonged to the compression decision, not to sizing."""
    from hermes_cli.context_switch_guard import _estimate_tokens

    cc = _compressor(monkeypatch, context_length=200_000)
    agent = SimpleNamespace(context_compressor=cc)

    assert _estimate_tokens(agent, [{"role": "user", "content": "x" * 40_000}])[0] > 1_000
    assert _estimate_tokens(agent, None) is None  # no history and no recorded counters


def test_no_cold_read_note_when_reselecting_the_current_route(monkeypatch):
    """`/model <the model the session already runs on>` lands on the route whose prefix cache is warm:
    the re-read never happens, so the note must not claim it. The warning merges before the swap on
    every surface, so ``agent.model`` is the current route; a provider change is a different route."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (90_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        model="large-model",
        base_url="",
        api_key="",
    )
    same_route = _result(model="large-model")
    merge_preflight_compression_warning(same_route, agent=agent)
    assert not same_route.warning_message

    same_model_other_provider = _result(model="large-model")
    same_model_other_provider.provider_changed = True
    merge_preflight_compression_warning(same_model_other_provider, agent=agent)
    assert "no warm prefix cache" in same_model_other_provider.warning_message


def test_endpoint_only_move_is_not_a_reselect(monkeypatch):
    """Same model + same provider label is not the same deployment: a direct-alias selection can
    move the session to another explicit URL, whose prefix cache is not the warm one."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (90_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        model="large-model",
        provider="openrouter",
        base_url="https://a.example/v1",
        api_key="",
    )
    moved = _result(model="large-model")
    moved.base_url = "https://b.example/v1"
    merge_preflight_compression_warning(moved, agent=agent)
    assert "no warm prefix cache" in moved.warning_message

    # Same URL stays a reselect: no invented warning there.
    same_url = _result(model="large-model")
    same_url.base_url = "https://a.example/v1"
    merge_preflight_compression_warning(same_url, agent=agent)
    assert not same_url.warning_message


def test_unanchored_target_does_not_promise_compression(monkeypatch):
    """Past the trigger without an anchored figure the switched compressor defers to the provider's
    real usage, so the summary states both dispositions instead of promising a pass."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (160_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        base_url="",
        api_key="",
    )
    result = _result(model="large-model")
    # Past protect_first_n + protect_last_n: a history the compressor could drop is a precondition
    # for the promise branch at all (a protected-length history gets the cold-read note instead).
    merge_preflight_compression_warning(
        result, agent=agent, messages=[{"role": "user", "content": "hi"} for _ in range(30)])
    assert "will run preflight compression" not in result.warning_message
    assert "either runs preflight compression" in result.warning_message


def test_anchored_target_keeps_the_compression_promise(monkeypatch):
    """The healthy same-route control: a valid usage anchor prices this transcript, the switch keeps
    the deployment the anchor was priced on (same model, provider, endpoint and wire), and preflight
    never defers on a priced figure — so the definite promise stands, read from the same anchor the
    runtime consults. The route-move complement lives in
    ``test_route_move_prices_the_display_but_not_the_promise``."""
    from agent.usage_anchor import capture_usage_anchor

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (160_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)
    messages = [{"role": "user", "content": "hi"} for _ in range(30)]
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        model="large-model",
        provider="openrouter",
        api_mode="chat_completions",
        base_url="https://example.com/v1",
        api_key="",
        _usage_anchor=capture_usage_anchor(160_000, 10, messages),
    )
    result = _result(model="large-model")
    merge_preflight_compression_warning(result, agent=agent, messages=messages)
    assert "will run preflight compression before the model replies" in result.warning_message


def test_codex_native_target_gets_the_cold_read_note(monkeypatch):
    """codex app-server compacts its own thread: Hermes runs no preflight pass there, so the
    summary must not promise one on a large session."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (160_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        api_mode="codex_app_server",
        codex_app_server_auto_compaction="native",
        base_url="",
        api_key="",
    )
    result = _result(model="large-model")
    merge_preflight_compression_warning(result, agent=agent)
    assert "no warm prefix cache" in result.warning_message
    assert "preflight compression" not in result.warning_message


def test_gateway_warns_from_durable_history_without_a_cached_agent(monkeypatch):
    """The switched session's cached agent is evicted before the next turn, so a second `/model`
    runs with no agent and the same large conversation. History, not residency, decides."""
    from threading import Lock

    from hermes_cli.context_switch_guard import enrich_model_switch_warnings_for_gateway

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    messages = [{"role": "user", "content": "x" * 20_000} for _ in range(24)]
    session = SimpleNamespace(session_id="s1", last_prompt_tokens=120_000)
    runner = SimpleNamespace(
        _agent_cache={},
        _agent_cache_lock=Lock(),
        _session_db=SimpleNamespace(get_messages_as_conversation=lambda sid: messages),
        session_store=SimpleNamespace(get_or_create_session=lambda source: session),
    )
    result = _result(model="large-model")
    enrich_model_switch_warnings_for_gateway(
        result, runner, session_key="k", source=object())
    assert "no warm prefix cache" in result.warning_message

    empty = SimpleNamespace(
        _agent_cache={},
        _agent_cache_lock=Lock(),
        _session_db=SimpleNamespace(get_messages_as_conversation=lambda sid: []),
        session_store=SimpleNamespace(
            get_or_create_session=lambda source: SimpleNamespace(session_id="s1", last_prompt_tokens=0)),
    )
    silent = _result(model="large-model")
    enrich_model_switch_warnings_for_gateway(silent, empty, session_key="k", source=object())
    assert not silent.warning_message


def test_structural_no_op_backoff_suppresses_the_compression_promise(monkeypatch):
    """The backoff a structural no-op arms is the one blocker a switch does not clear —
    ``update_model`` resets the strikes and the summary cooldown, never
    ``_structural_no_op_backoff_until`` — so the summary must follow the engine's own admission
    rather than a blocker list re-derived here."""
    from agent.usage_anchor import capture_usage_anchor

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (160_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    messages = [{"role": "user", "content": "hi"} for _ in range(30)]
    cc._record_structural_no_op("nothing eligible")
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        model="large-model",
        provider="openrouter",
        base_url="https://a.example/v1",
        api_key="",
        _usage_anchor=capture_usage_anchor(160_000, 10, messages),
    )
    result = _result(model="large-model")
    result.base_url = "https://b.example/v1"

    merge_preflight_compression_warning(result, agent=agent, messages=messages)

    # The runtime refuses the pass at this figure; the summary must not promise it.
    assert cc.should_compress_info(160_000)[1].startswith("structural_backoff:")
    assert "will run preflight compression" not in result.warning_message
    assert "no warm prefix cache" in result.warning_message


def test_label_only_alias_swap_is_not_a_route_change(monkeypatch):
    """Deployment facts run before the label: two custom aliases at one URL + model are one
    deployment, whose prefix cache is the warm one; a moved URL under either label is another."""
    from agent.backend_identity import BackendIdentity, same_deployment

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (90_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        model="large-model",
        provider="custom:a",
        base_url="https://old.example/v1",
        api_key="",
    )
    old_url, new_url = "https://old.example/v1", "https://new.example/v1"
    oracle = same_deployment(
        BackendIdentity.build(provider="custom:a", model="large-model", base_url=old_url),
        BackendIdentity.build(provider="custom:b", model="large-model", base_url=old_url))

    alias = _result(model="large-model")
    alias.target_provider = "custom:b"
    alias.provider_changed = True
    alias.base_url = old_url
    assert oracle  # the identity owner calls this one deployment
    merge_preflight_compression_warning(alias, agent=agent)
    assert not alias.warning_message

    moved = _result(model="large-model")
    moved.target_provider = "custom:b"
    moved.provider_changed = True
    moved.base_url = new_url
    assert not same_deployment(
        BackendIdentity.build(provider="custom:a", model="large-model", base_url=old_url),
        BackendIdentity.build(provider="custom:b", model="large-model", base_url=new_url))
    merge_preflight_compression_warning(moved, agent=agent)
    assert "no warm prefix cache" in moved.warning_message


def test_durable_prompt_tokens_size_a_session_with_no_transcript(monkeypatch):
    """Wired callers pass a list, never ``None``: an empty transcript is "no evidence", so the
    session row's own figure still sizes the payload, and a real transcript outranks it. That figure
    is a *record* of an earlier request, so the note it drives names it as one and says the current
    request size is unavailable — history is never this turn's payload."""
    from hermes_cli.context_switch_guard import _estimate_tokens

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    agent = SimpleNamespace(context_compressor=cc, compression_enabled=True, base_url="", api_key="")

    assert _estimate_tokens(agent, [], 180_000) == (180_000, "record", False)
    assert _estimate_tokens(agent, [], 0) is None
    assert _estimate_tokens(agent, [{"role": "user", "content": "x" * 40_000}], 20_000)[0] != 20_000

    result = _result(model="large-model")
    merge_preflight_compression_warning(
        result, agent=agent, messages=[], durable_prompt_tokens=180_000)
    assert "Last recorded prompt size was ~180,000 tokens" in result.warning_message
    assert "the current request size is unavailable" in result.warning_message
    assert "no warm prefix cache" in result.warning_message
    # The historical figure carries the read, never the size: no claim about *this* payload.
    assert "Session is ~" not in result.warning_message
    assert "re-reads them" not in result.warning_message


def test_window_shrink_blocker_verdict_is_read_at_the_target_trigger(monkeypatch):
    """The blocker check has to be made against the target the switch installs. On a 1M → 200K move
    the quoted figure sits below the trigger the old compressor still owns, so ``should_compress_info``
    reports no reason there while the target turn prices a lower trigger that the same figure is past:
    a block found by that preview is that blocker's state, not an admission from the target."""
    from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (165_255, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    messages = [{"role": "user", "content": "hi"} for _ in range(30)]
    anchor = capture_usage_anchor(160_000, 10, messages)
    priced = anchored_context_tokens(messages, anchor)

    def _agent(cc):
        return SimpleNamespace(
            context_compressor=cc,
            compression_enabled=True,
            model="large-model",
            provider="openrouter",
            base_url="https://a.example/v1",
            api_key="",
            _usage_anchor=anchor,
        )

    # Old window 1M (trigger 500k), target 200k (trigger 150k), priced 160,010: past the target
    # trigger, below the old one — with a live structural backoff the pass is not promised.
    blocked = _compressor(monkeypatch, context_length=1_000_000)
    blocked._record_structural_no_op("nothing eligible")
    result = _result(model="large-model")
    result.base_url = "https://b.example/v1"
    merge_preflight_compression_warning(result, agent=_agent(blocked), messages=messages)
    assert "will run preflight compression" not in result.warning_message
    assert "either runs preflight compression" in result.warning_message
    # The target-side verdict the wording has to agree with: the switch keeps the backoff.
    blocked.update_model("large-model", 200_000, provider="openrouter")
    assert blocked.should_compress_info(priced)[1].startswith("structural_backoff:")

    # Same move without a backoff: the engine would run the pass, but the anchor was priced on the
    # route this switch leaves and its authority over the destination is not established, so the
    # forecast stays conditional. Composition, in order: the warning, the switch, the preflight.
    healthy = _compressor(monkeypatch, context_length=1_000_000)
    result2 = _result(model="large-model")
    result2.base_url = "https://b.example/v1"
    merge_preflight_compression_warning(result2, agent=_agent(healthy), messages=messages)
    assert "will run preflight compression" not in result2.warning_message
    assert "either runs preflight compression" in result2.warning_message
    healthy.update_model("large-model", 200_000, provider="openrouter")
    assert healthy.should_compress_info(priced) == (True, None)  # the run really does take the pass


def test_sub_trigger_anchor_is_priced_by_the_runtime_not_the_rough_estimate(monkeypatch):
    """A valid anchor is the figure the next turn prices, so it is both what the promise is decided
    on and what the user is quoted: an anchor below the target trigger is a turn that needs no pass,
    however large the raw transcript sizes out (the complement — anchor above the trigger — keeps the
    promise, pinned by ``test_anchored_target_keeps_the_compression_promise``)."""
    from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (165_255, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    messages = [{"role": "user", "content": "hi"} for _ in range(30)]
    anchor = capture_usage_anchor(80_000, 10, messages)
    priced = anchored_context_tokens(messages, anchor)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        model="large-model",
        provider="openrouter",
        base_url="https://a.example/v1",
        api_key="",
        _usage_anchor=anchor,
    )
    result = _result(model="large-model")
    result.base_url = "https://b.example/v1"

    merge_preflight_compression_warning(result, agent=agent, messages=messages)

    assert priced < cc.threshold_tokens and cc.should_compress_info(priced)[0] is False
    assert "will run preflight compression" not in result.warning_message
    assert f"{priced:,} tokens" in result.warning_message
    assert "165,255" not in result.warning_message  # the stale rough figure is not what it prices


def test_codex_app_server_entry_does_not_promise_a_hermes_pass(monkeypatch):
    """``agent.api_mode`` is still the OLD wire when the warning is merged, so reading it alone
    describes where the session came from: a switch INTO codex app-server promised a Hermes preflight
    pass that codex owns instead."""
    from agent.turn_context_compaction import _codex_native_auto_compaction

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (160_000, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        model="small-model",
        provider="openrouter",
        api_mode="chat_completions",
        base_url="https://a.example/v1",
        api_key="",
    )
    result = _result(model="large-model")
    result.api_mode = "codex_app_server"
    assert _codex_native_auto_compaction(SimpleNamespace(api_mode="codex_app_server"))

    merge_preflight_compression_warning(result, agent=agent)

    assert "preflight compression" not in result.warning_message
    assert "no warm prefix cache" in result.warning_message


def test_gateway_durable_figure_sizes_a_session_with_no_transcript(monkeypatch):
    """The row's own API-reported prompt size is the evidence this path exists to supply: a session
    with no transcript rows yet still has a size to quote, and "no transcript" is not "no evidence"."""
    from threading import Lock

    from hermes_cli.context_switch_guard import enrich_model_switch_warnings_for_gateway

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    drained = SimpleNamespace(
        _agent_cache={},
        _agent_cache_lock=Lock(),
        _session_db=SimpleNamespace(get_messages_as_conversation=lambda sid: []),
        session_store=SimpleNamespace(
            get_or_create_session=lambda source: SimpleNamespace(session_id="s1", last_prompt_tokens=180_000)),
    )
    result = _result(model="large-model")
    enrich_model_switch_warnings_for_gateway(result, drained, session_key="k", source=object())
    assert "180,000 tokens" in result.warning_message

    nothing = SimpleNamespace(
        _agent_cache={},
        _agent_cache_lock=Lock(),
        _session_db=SimpleNamespace(get_messages_as_conversation=lambda sid: []),
        session_store=SimpleNamespace(
            get_or_create_session=lambda source: SimpleNamespace(session_id="s1", last_prompt_tokens=0)),
    )
    silent = _result(model="large-model")
    enrich_model_switch_warnings_for_gateway(silent, nothing, session_key="k", source=object())
    assert not silent.warning_message


def test_session_counter_is_labelled_as_a_total_not_a_size(monkeypatch):
    """``session_prompt_tokens`` sums every request the session ever sent, so it is neither the last
    measured request nor this turn's payload. Paired with the compressor's own provider reading,
    which keeps the size wording — the reading has to come from ``update_from_response``, because a
    bare ``last_prompt_tokens`` assignment is the display seed the neighbouring test pins."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        session_prompt_tokens=170_000,
        base_url="",
        api_key="",
    )

    counting = _result(model="large-model")
    merge_preflight_compression_warning(counting, agent=agent, messages=[])
    assert "This session has sent ~170,000 prompt tokens in total" in counting.warning_message
    assert "the current request size is unavailable" in counting.warning_message
    assert "Session is ~" not in counting.warning_message
    # A cumulative counter is not the durable record: the two fallbacks stay distinguishable.
    assert "Last recorded prompt size" not in counting.warning_message

    cc.update_from_response({"prompt_tokens": 170_000, "completion_tokens": 0})
    measured = _result(model="large-model")
    merge_preflight_compression_warning(measured, agent=agent, messages=[])
    assert "Session is ~170,000 tokens" in measured.warning_message
    assert "unavailable" not in measured.warning_message


def test_display_seed_is_a_local_estimate_not_a_measurement(monkeypatch):
    """``maybe_seed_preflight_display_tokens`` fills ``last_prompt_tokens`` from a local estimate
    while ``last_real_prompt_tokens`` stays 0. That figure sizes the session for display; nothing
    provider-side has priced it, so it is named as an estimate and never quoted as "the session is
    ~N tokens", and the copy does not claim to have measured the payload it re-reads. The real
    reading for the same figure takes the size wording back."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    cc.maybe_seed_preflight_display_tokens(170_000)
    assert cc.last_prompt_tokens == 170_000 and cc.last_real_prompt_tokens == 0
    agent = SimpleNamespace(context_compressor=cc, compression_enabled=True, base_url="", api_key="")

    seeded = _result(model="large-model")
    merge_preflight_compression_warning(seeded, agent=agent, messages=[])
    message = seeded.warning_message
    assert "Estimated current request size ~170,000 tokens" in message
    assert "(local estimate, no provider reading for it yet)" in message
    assert "Session is ~" not in message
    assert "re-reads them" not in message

    cc.update_from_response({"prompt_tokens": 170_000, "completion_tokens": 0})
    assert cc.last_real_prompt_tokens == 170_000
    measured = _result(model="large-model")
    merge_preflight_compression_warning(measured, agent=agent, messages=[])
    assert "Session is ~170,000 tokens" in measured.warning_message
    assert "local estimate" not in measured.warning_message


def test_historical_figure_cannot_locate_the_next_request(monkeypatch):
    """F1: a figure that cannot price the current request cannot say the next message is past the
    trigger either. The sentence may quote the record it read and the trigger the switch installs;
    the comparison between them is unknown until a provider reading arrives."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    agent = SimpleNamespace(context_compressor=cc, compression_enabled=True, base_url="", api_key="")

    historical = _result(model="large-model")
    merge_preflight_compression_warning(
        historical, agent=agent, messages=None, durable_prompt_tokens=180_000)
    assert "the current request size is unavailable" in historical.warning_message
    assert "Your next message is past that trigger" not in historical.warning_message
    assert "Whether the next message is past that trigger is unknown" in historical.warning_message
    assert "either runs preflight compression" in historical.warning_message

    # The control: a provider reading past the same trigger may state the comparison.
    cc.update_from_response({"prompt_tokens": 180_000, "completion_tokens": 0})
    measured = _result(model="large-model")
    merge_preflight_compression_warning(measured, agent=agent, messages=None)
    assert "Your next message is past that trigger" in measured.warning_message
    assert "Whether the next message is past that trigger is unknown" not in measured.warning_message


def test_returning_to_a_route_is_not_claimed_as_first_service(monkeypatch):
    """F4: ``_same_route`` compares the deployment the session is on *now*, and nothing the guard can
    read records whether a destination served the session earlier — an A→B→A session returning to A
    has warmed A's cache once already. So no branch may state first service as fact; the cold cache
    is stated as the condition that would make the read expensive."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    cc.update_model(
        "model-b", 200_000, provider="openrouter",
        base_url="https://b.example/v1", api_mode="chat_completions")
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=False,  # no compression promise in the way
        model="model-b",
        provider="openrouter",
        base_url="https://b.example/v1",
        api_key="",
        api_mode="chat_completions",
    )

    returned = _result(model="model-a")
    returned.base_url = "https://a.example/v1"
    merge_preflight_compression_warning(
        returned, agent=agent, messages=[], durable_prompt_tokens=120_000)

    message = returned.warning_message
    assert "no warm prefix cache" in message
    assert "model-a has not served this session" not in message
    assert "if that route has not served this session" in message

    # The measured branch carries the same condition rather than the same claim.
    cc.update_from_response({"prompt_tokens": 120_000, "completion_tokens": 0})
    measured = _result(model="model-a")
    measured.base_url = "https://a.example/v1"
    merge_preflight_compression_warning(measured, agent=agent, messages=[], durable_prompt_tokens=120_000)
    assert "model-a has not served this session" not in measured.warning_message
    assert "if that route has not served this session" in measured.warning_message



def test_historical_figure_never_promises_an_eventual_compression(monkeypatch):
    """Above the trigger with only the session row's recorded figure (a caller with no transcript at
    all): the sentence says which record it read, says the current request size is unavailable, and
    promises neither a pass now nor one later — a fresh provider reading informs the next decision,
    it is not an outcome this reading guarantees."""
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    agent = SimpleNamespace(context_compressor=cc, compression_enabled=True, base_url="", api_key="")
    result = _result(model="large-model")
    merge_preflight_compression_warning(
        result, agent=agent, messages=None, durable_prompt_tokens=180_000)

    message = result.warning_message
    assert "Last recorded prompt size was ~180,000 tokens" in message
    assert "the current request size is unavailable" in message
    assert "Session is ~" not in message and "re-reads them" not in message
    assert "will run preflight compression" not in message
    assert "either runs preflight compression" in message
    # The promise this wording replaced: an eventual pass the reading alone cannot guarantee.
    assert "compresses once the provider reports real usage" not in message


def test_route_move_prices_the_display_but_not_the_promise(monkeypatch):
    """An anchor is priced on the route the session runs on and carries no route of its own, so only a
    destination it can speak for gets a definite promise — a move to another deployment (or another
    wire on one deployment) may quote the figure but never promises on it. Paired with the same-route
    control, which keeps its supported behaviour, and composed end to end: warning → switch →
    preflight."""
    from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (165_255, "measured", True),
    )
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000,
    )
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 150k
    messages = [{"role": "user", "content": "hi"} for _ in range(30)]
    anchor = capture_usage_anchor(160_000, 10, messages)
    priced = anchored_context_tokens(messages, anchor)
    assert priced >= cc.threshold_tokens  # the anchor itself would reach the trigger
    agent = SimpleNamespace(
        context_compressor=cc,
        compression_enabled=True,
        model="large-model",
        provider="openrouter",
        api_mode="chat_completions",
        base_url="https://a.example/v1",
        api_key="",
        _usage_anchor=anchor,
    )

    # Paired control: the deployment the anchor was priced on keeps the definite promise.
    same_route = _result(model="large-model")
    same_route.base_url = "https://a.example/v1"
    merge_preflight_compression_warning(same_route, agent=agent, messages=messages)
    assert "will run preflight compression before the model replies" in same_route.warning_message

    # Another deployment: the same anchor prices the display, it does not authorize the promise.
    moved = _result(model="large-model")
    moved.base_url = "https://b.example/v1"
    merge_preflight_compression_warning(moved, agent=agent, messages=messages)
    assert f"{priced:,} tokens" in moved.warning_message
    assert "will run preflight compression" not in moved.warning_message
    assert "either runs preflight compression" in moved.warning_message

    # Another wire on one deployment is another route for the prefix cache, and not the same read.
    other_wire = _result(model="large-model")
    other_wire.base_url = "https://a.example/v1"
    other_wire.api_mode = "responses"
    merge_preflight_compression_warning(other_wire, agent=agent, messages=messages)
    assert "will run preflight compression" not in other_wire.warning_message
    assert "either runs preflight compression" in other_wire.warning_message

    # Composition: the run then switches and does take the pass the forecast declined to promise —
    # reported here rather than inferred from the isolated-head cases above.
    cc.update_model("large-model", 200_000, provider="openrouter")
    assert cc.should_compress_info(priced) == (True, None)


def test_agentless_trigger_agrees_with_the_engine_it_would_install(monkeypatch):
    """F3: with no live engine the guard still quotes the trigger the switch installs. It reads the
    same ``compression`` config section the engine is built from and runs the engine's own math, so
    a resident and an evicted caller describe one policy one way — including a per-model override
    and an absolute cap, which a ratio invented at the call site would miss."""
    from agent.context_compressor import ContextCompressor

    from hermes_cli import context_switch_guard as guard

    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *a, **k: 200_000)

    configured = {
        "threshold": 0.4,
        "model_thresholds": {"large": 0.6},
        "threshold_tokens": 130_000,
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"compression": configured})
    engine = ContextCompressor(
        model="big-model", threshold_percent=0.4, quiet_mode=True,
        config_context_length=200_000, model_thresholds={"large": 0.6},
        threshold_tokens_cap=130_000)

    for model in ("large-model", "other-model"):
        assert guard._threshold_tokens(None, model, 200_000, "openrouter") == \
            engine.preview_threshold_tokens(model, 200_000, "openrouter")

    # An unreadable/empty config still states the shipped policy, and still agrees with the engine
    # built from it.
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    plain = ContextCompressor(model="big-model", quiet_mode=True, config_context_length=200_000)
    assert guard._threshold_tokens(None, "large-model", 200_000, "openrouter") == \
        plain.preview_threshold_tokens("large-model", 200_000, "openrouter")


def test_endpoint_move_is_one_transition_on_both_surfaces(monkeypatch):
    """F5: the switch summary and the selection confirmation read one resolved transition.

    Same model, endpoint moved A→B while the session holds 200,000 provider-counted tokens: the
    summary emits the cold-read note, so the confirmation has to ask — it used to answer from the
    model string alone and stayed silent, describing that one transition two ways. A reselect of the
    deployment the session already runs on, including a new alias label at the same URL and model, is
    silent on both surfaces.
    """
    from hermes_cli.model_selection_guards import SelectionContext, _context_cache_guard

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens", lambda *a, **k: (200_000, "measured", True))
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length", lambda *a, **k: 500_000)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    cc = _compressor(monkeypatch, context_length=500_000)

    def both(current_url, target_url, current_provider="openrouter", target_provider="openrouter"):
        agent = SimpleNamespace(
            context_compressor=cc, compression_enabled=True, model="model-a",
            provider=current_provider, base_url=current_url, api_key="", api_mode="chat_completions")
        result = _result(model="model-a")
        result.target_provider = target_provider
        result.base_url = target_url
        merge_preflight_compression_warning(result, agent=agent)
        ctx = SelectionContext(
            context_tokens=200_000, current_model="model-a",
            current_provider=current_provider, current_base_url=current_url)
        return result.warning_message or "", _context_cache_guard(
            "model-a", target_provider, target_url, None, None, ctx)

    moved, confirmed_move = both("https://a.example/v1", "https://b.example/v1")
    assert "no warm prefix cache" in moved
    assert confirmed_move is not None, "the confirmation dropped a move the summary warned about"

    reselect, confirmed_reselect = both("https://a.example/v1", "https://a.example/v1")
    assert "no warm prefix cache" not in reselect
    assert confirmed_reselect is None

    aliased, confirmed_alias = both(
        "https://a.example/v1", "https://a.example/v1", "custom:a", "custom:b")
    assert "no warm prefix cache" not in aliased
    assert confirmed_alias is None


def test_destination_projection_strips_stale_reasoning_the_target_wire_drops(monkeypatch):
    """R1: the summary sizes the request the *destination* reads, not the stored bytes.

    A reasoning-heavy transcript moved to a route whose wire never replays stale thinking
    (openai/chat_completions) costs a fraction of what the stored transcript holds, and the
    runtime preflight reads the same figure — the summary must not claim the destination
    request is past the trigger. Paired controls: the same transcript on a route that replays
    the thinking (deepseek echo family) stays over the trigger and warns, and a content-heavy
    transcript stays over it on the stripping route.
    """
    from agent.turn_context import RequestRoute, _preflight_request_tokens

    from hermes_cli.context_switch_guard import _estimate_tokens

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000)
    cc = _compressor(monkeypatch, context_length=200_000)  # trigger at 100k
    agent = SimpleNamespace(
        context_compressor=cc, compression_enabled=True, model="origin-model",
        provider="openrouter", api_mode="chat_completions",
        base_url="https://a.example/v1", api_key="", tools=[], _cached_system_prompt="")
    reasoning = []
    for i in range(15):
        reasoning.extend([
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer",
             "reasoning_content": "r" * 64_000 if i < 14 else "brief"}])

    def assess(target_model, target_provider, target_url):
        result = _result(model=target_model)
        result.target_provider = target_provider
        result.base_url = target_url
        merge_preflight_compression_warning(result, agent=agent, messages=reasoning)
        return result.warning_message or ""

    openai_route = RequestRoute(
        model="gpt-4.1", provider="openai", base_url="https://api.openai.com/v1",
        api_mode="chat_completions")
    projected = _estimate_tokens(agent, reasoning, route=openai_route)
    assert projected.source == "estimate" and projected.applies
    assert projected.figure < cc.threshold_tokens

    # The runtime reads the same projection on the destination route: no drift between the
    # figure the warning quotes and the figure preflight compares.
    post_switch = SimpleNamespace(
        model="gpt-4.1", provider="openai", api_mode="chat_completions",
        base_url="https://api.openai.com/v1", tools=[], _cached_system_prompt="",
        _usage_anchor=None)
    assert projected.figure == _preflight_request_tokens(post_switch, reasoning, "")

    # Below the trigger on the destination wire: no size and no threshold claim.
    openai = assess("gpt-4.1", "openai", "https://api.openai.com/v1")
    assert "past that trigger" not in openai
    assert "no warm prefix cache" not in openai  # below half the trigger too

    # Paired control: a route that replays the stale thinking keeps the same transcript over
    # the trigger and warns.
    echoed = assess("deepseek-reasoner", "deepseek", "https://api.deepseek.com/v1")
    assert "Your next message is past that trigger" in echoed

    # Paired control: content-heavy bytes stay on every wire and stay over the trigger.
    content = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "r" * 64_000}
        for i in range(30)
    ]
    result = _result(model="gpt-4.1")
    result.target_provider = "openai"
    result.base_url = "https://api.openai.com/v1"
    merge_preflight_compression_warning(result, agent=agent, messages=content)
    assert "Your next message is past that trigger" in result.warning_message


def test_prior_route_reading_keeps_its_weaker_class(monkeypatch):
    """R1: the anchor override may not promote a reading taken on the session's own route.

    The usage anchor is the provider's priced figure for this transcript, captured on the route
    the session runs on and carrying no route of its own; a switch to another deployment keeps
    it as a *prior-route* reading — the figure stays visible, but as the weaker evidence class:
    it never quotes a size for the target's request and never locates that request against the
    trigger. Paired control: the same anchor on its own route stays measured and keeps the
    definite promise.
    """
    from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length",
        lambda *a, **k: 200_000)
    cc = _compressor(monkeypatch, context_length=200_000)
    messages = [{"role": "user", "content": "hi"} for _ in range(30)]
    anchor = capture_usage_anchor(160_000, 10, messages)
    priced = anchored_context_tokens(messages, anchor)
    agent = SimpleNamespace(
        context_compressor=cc, compression_enabled=True, model="large-model",
        provider="openrouter", api_mode="chat_completions",
        base_url="https://a.example/v1", api_key="", _usage_anchor=anchor)

    moved = _result(model="large-model")
    moved.base_url = "https://b.example/v1"
    merge_preflight_compression_warning(moved, agent=agent, messages=messages)
    message = moved.warning_message
    assert f"Last provider reading on this session's route was ~{priced:,} tokens" in message
    assert "it does not price the target's request" in message
    assert "Session is ~" not in message
    assert "Your next message is past that trigger" not in message
    assert "Whether the next message is past that trigger is unknown" in message

    same = _result(model="large-model")
    same.base_url = "https://a.example/v1"
    merge_preflight_compression_warning(same, agent=agent, messages=messages)
    assert "will run preflight compression before the model replies" in same.warning_message


def test_native_responses_projection_is_one_owner_for_both_consumers(monkeypatch):
    """The checkpoint-pruned native projection is shared, not re-derived per consumer.

    On an eligible native Responses route both the runtime preflight and the switch assessment
    quote the pruned figure the native owner produces; off that route both fall back to the
    generic transcript estimate for the same transcript. Paired so neither consumer can drift
    alone.
    """
    from agent.turn_context import (
        RequestRoute, _preflight_request_tokens, project_request_pressure)

    from hermes_cli.context_switch_guard import _estimate_tokens

    messages = [{"role": "user", "content": "x" * 80_000} for _ in range(4)]
    monkeypatch.setattr(
        "agent.codex_responses_adapter.estimate_native_responses_preflight_tokens",
        lambda agent, messages, **k: 5_000)
    native_route = RequestRoute(
        model="gpt-5-codex", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses")

    assert project_request_pressure(
        native_route, messages,
        native_agent=SimpleNamespace(api_mode="codex_responses")) == (5_000, "native")

    codex_agent = SimpleNamespace(
        api_mode="codex_responses", model="gpt-5-codex", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", tools=None,
        _cached_system_prompt="", _usage_anchor=None)
    assert _preflight_request_tokens(codex_agent, messages, "") == 5_000

    agent = SimpleNamespace(
        context_compressor=None, model="origin-model", provider="openrouter",
        api_mode="chat_completions", base_url="https://a.example/v1", tools=None,
        _cached_system_prompt="")
    assert _estimate_tokens(agent, messages, route=native_route) == (5_000, "estimate", True)

    generic = _estimate_tokens(agent, messages, route=RequestRoute(
        model="gpt-4.1", provider="openai", base_url="https://api.openai.com/v1",
        api_mode="chat_completions"))
    assert generic.figure > 5_000 and generic.source == "estimate"
    plain_agent = SimpleNamespace(
        api_mode="chat_completions", model="gpt-4.1", provider="openai",
        base_url="https://api.openai.com/v1", tools=None, _cached_system_prompt="",
        _usage_anchor=None)
    assert _preflight_request_tokens(plain_agent, messages, "") == generic.figure


def test_both_surfaces_read_one_evidence_owner(monkeypatch):
    """The switch summary and the selection confirmation size a session through one owner.

    One compressor state yields one evidence class and one rendering on both surfaces: the
    display seed is an estimate, a provider reading is a measurement, the session counter is
    a total — summary class, confirmation class and the quoted words agree in each state.
    """
    from hermes_cli.context_switch_guard import _estimate_tokens, size_label
    from hermes_cli.model_selection_guards import selection_context_for_agent

    cc = _compressor(monkeypatch, context_length=200_000)
    agent = SimpleNamespace(
        context_compressor=cc, compression_enabled=True, model="big-model",
        provider="openrouter", base_url="", api_key="")

    cc.maybe_seed_preflight_display_tokens(170_000)
    sized = _estimate_tokens(agent, None)
    ctx = selection_context_for_agent(agent)
    assert (sized.source, sized.applies) == ("estimate", False)
    assert (ctx.context_tokens, ctx.context_tokens_source) == (170_000, "estimate")
    assert size_label(sized.figure, sized.source) == size_label(
        ctx.context_tokens, ctx.context_tokens_source)

    cc.update_from_response({"prompt_tokens": 170_000, "completion_tokens": 0})
    sized = _estimate_tokens(agent, None)
    ctx = selection_context_for_agent(agent)
    assert (sized.source, sized.applies) == ("measured", True)
    assert (ctx.context_tokens, ctx.context_tokens_source) == (170_000, "measured")
    assert size_label(sized.figure, sized.source) == size_label(
        ctx.context_tokens, ctx.context_tokens_source)

    fresh = _compressor(monkeypatch, context_length=200_000)
    counter_agent = SimpleNamespace(
        context_compressor=fresh, compression_enabled=True, model="big-model",
        provider="openrouter", base_url="", api_key="", session_prompt_tokens=170_000)
    sized = _estimate_tokens(counter_agent, None)
    ctx = selection_context_for_agent(counter_agent)
    assert (sized.source, sized.applies) == ("counter", False)
    assert (ctx.context_tokens, ctx.context_tokens_source) == (170_000, "counter")
    assert size_label(sized.figure, sized.source) == size_label(
        ctx.context_tokens, ctx.context_tokens_source)


# ── The projection's *inputs* are destination-owned too (N1/N2 residuals) ─────────────────────
# The rebuild centralized the projection at the function name but left two inputs source-owned: the
# native gate read the runtime the session was leaving, and the installed-policy producer's
# "external engine owns this" sentinel was consumed as "could not answer".


def _native_checkpoint_history():
    """A durable transcript whose pre-checkpoint rows drop off an eligible native wire (#96155)."""
    pre = []
    for i in range(30):
        pre.append({"role": "user", "content": f"ask {i}"})
        pre.append({"role": "assistant", "content": "working " + ("tool output " * 400)})
        pre.append({"role": "tool", "content": "result " + ("payload " * 400),
                    "tool_call_id": f"call-{i}"})
    checkpoint = {
        "role": "assistant",
        "content": "checkpointed turn",
        "codex_reasoning_items": [
            {"type": "compaction", "encrypted_content": "blob", "_issuer_kind": "codex_backend"}
        ],
    }
    return pre + [checkpoint, {"role": "user", "content": "follow-up after checkpoint"}]


def test_switch_projection_gates_on_the_destination_capability_map(monkeypatch):
    """N1: the switch projection reads the *destination* capability map, not the source runtime's.

    ``switch_model`` publishes the resolved target map (``agent.runtime_capabilities =
    destination_capabilities``) and the native gate reads it. A view that kept the source runtime's
    ``native_compaction: False`` priced the durable transcript generically in the warning while the
    next real preflight checkpoint-pruned the same session — one projector, two runtime inputs.
    """
    from agent.codex_responses_adapter import estimate_native_responses_preflight_tokens
    from agent.native_compaction import native_compaction_context_management
    from agent.turn_context import RequestRoute, _preflight_request_tokens

    from hermes_cli import context_switch_guard as guard

    messages = _native_checkpoint_history()
    codex_route = RequestRoute(
        model="gpt-5.6", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses")
    source = SimpleNamespace(
        api_mode="chat_completions", model="origin-model", provider="openrouter",
        base_url="https://a.example/v1", tools=None, _cached_system_prompt="",
        runtime_capabilities={"native_compaction": False},
        codex_responses_native_compaction=True, compression_enabled=True,
        compression_checkpoint_required=False, _codex_reasoning_replay_enabled=True,
        codex_responses_compact_threshold=None, capabilities={},
        context_compressor=SimpleNamespace(threshold_tokens=150_000))

    # The source runtime's own map closes the gate, exactly as the live agent would.
    assert native_compaction_context_management(
        guard._destination_native_view(source, codex_route), is_codex_backend=True) is None

    # The resolved destination map opens it, and the whole projection moves with it.
    view = guard._destination_native_view(
        source, codex_route, runtime_capabilities={"native_compaction": True})
    assert native_compaction_context_management(view, is_codex_backend=True) is not None

    switched = SimpleNamespace(**vars(source))
    switched.model, switched.provider, switched.base_url, switched.api_mode = (
        codex_route.model, codex_route.provider, codex_route.base_url, codex_route.api_mode)
    switched.runtime_capabilities = {"native_compaction": True}
    pruned = estimate_native_responses_preflight_tokens(switched, messages)
    assert pruned is not None and pruned < 8_000

    # Both consumers select the checkpoint-pruned figure: the runtime preflight and the warning.
    assert _preflight_request_tokens(switched, messages, "") == pruned
    sized = guard._estimate_tokens(
        source, messages, route=codex_route, runtime_capabilities={"native_compaction": True})
    assert sized is not None and sized.figure == pruned

    # Reverse control: without the capability on the destination neither consumer prunes.
    assert native_compaction_context_management(
        guard._destination_native_view(
            source, codex_route, runtime_capabilities={"native_compaction": False}),
        is_codex_backend=True) is None
    aged = SimpleNamespace(**vars(switched))
    aged.runtime_capabilities = {"native_compaction": False}
    unpruned = _preflight_request_tokens(aged, messages, "")
    assert unpruned > pruned * 2
    fallback = guard._estimate_tokens(
        source, messages, route=codex_route, runtime_capabilities={"native_compaction": False})
    assert fallback is not None and fallback.figure == unpruned


class _ExternalEngine:
    """A live external context engine: its own trigger, no preview contract, its own admission."""

    def __init__(self, threshold_tokens: int = 180_000, context_length: int = 200_000):
        self.threshold_tokens = threshold_tokens
        self.context_length = context_length

    def should_compress(self, tokens=None) -> bool:
        return int(tokens or 0) >= self.threshold_tokens

    def should_compress_info(self, tokens=None):
        return self.should_compress(tokens), None


def test_external_engine_answers_for_its_own_trigger(monkeypatch):
    """N2: a ``None`` from the installed-policy producer means "an external engine owns this".

    ``resolve_installed_compression_policy`` returns ``None`` when ``context.engine`` selects a
    plugin — the same value it returns when its producers cannot answer — and the guard read it as
    "unresolved" and quoted the built-in ratio instead: a 150K trigger and a definite pass forecast
    for an engine whose own admission is false at the priced figure.
    """
    from agent.usage_anchor import anchored_context_tokens, capture_usage_anchor

    from hermes_cli import context_switch_guard as guard

    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (160_000, "measured", True))
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length", lambda *a, **k: 200_000)
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._compression_config", lambda: {"threshold": 0.5})
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"compression": {"threshold": 0.5}})

    engine = _ExternalEngine()
    agent = SimpleNamespace(
        context_compressor=engine, compression_enabled=True, model="big-model",
        provider="openrouter", api_mode="chat_completions", base_url="https://a.example/v1",
        api_key="")

    assert guard._threshold_tokens(engine, "big-model", 200_000, "openrouter") == 180_000
    assert engine.should_compress(160_000) is False  # the engine's own admission at the figure

    # A moved deployment: the engine's own 180K trigger is above the priced figure, so the switch
    # costs a re-read — never the invented 150K trigger or a compression promise.
    moved = _result(model="big-model")
    moved.base_url = "https://b.example/v1"
    merge_preflight_compression_warning(
        moved, agent=agent, messages=[{"role": "user", "content": "x"}])
    assert "auto-compress at ~" not in moved.warning_message
    assert "will run preflight compression" not in moved.warning_message
    assert "re-reads them before it answers" in moved.warning_message

    # Same deployment with a provider reading above the built-in ratio but below the engine's own
    # trigger: still no pass promised. Paired control for the built-in compressor follows.
    messages = [{"role": "user", "content": "hi"} for _ in range(30)]
    anchor = capture_usage_anchor(160_000, 10, messages)
    priced = anchored_context_tokens(messages, anchor)
    anchored_agent = SimpleNamespace(**vars(agent))
    anchored_agent._usage_anchor = anchor
    same = _result(model="big-model")
    same.base_url = "https://a.example/v1"
    merge_preflight_compression_warning(same, agent=anchored_agent, messages=messages)
    assert priced < 180_000 and same.warning_message == ""

    builtin = _compressor(monkeypatch, context_length=200_000)
    builtin_agent = SimpleNamespace(**vars(agent))
    builtin_agent.context_compressor = builtin
    control = _result(model="big-model")
    control.base_url = "https://b.example/v1"
    merge_preflight_compression_warning(control, agent=builtin_agent, messages=messages)
    builtin_trigger = guard._threshold_tokens(builtin, "big-model", 200_000, "openrouter")
    assert builtin_trigger == builtin.preview_threshold_tokens("big-model", 200_000, "openrouter")
    assert f"auto-compress at ~{builtin_trigger:,}" in control.warning_message


def test_agentless_external_engine_destination_quotes_no_builtin_trigger(monkeypatch):
    """N2 agentless: an external engine owns the config and there is no live engine to ask, so no
    built-in figure describes the destination's first turn and none is invented."""
    from hermes_cli import context_switch_guard as guard

    monkeypatch.setattr(
        "agent.agent_init._select_context_engine", lambda cfg: SimpleNamespace(name="engine"))
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: {"compression": {"threshold": 0.5}})
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._estimate_tokens",
        lambda *a, **k: (160_000, "measured", True))
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard.resolve_display_context_length", lambda *a, **k: 200_000)
    monkeypatch.setattr(
        "hermes_cli.context_switch_guard._compression_config", lambda: {"threshold": 0.5})

    assert guard._resolved_installed_policy("big-model", "openrouter", 200_000) is None
    assert guard._threshold_tokens(None, "big-model", 200_000, "openrouter") is None

    result = _result(model="big-model")
    agent = SimpleNamespace(
        context_compressor=None, compression_enabled=True, model="big-model",
        provider="openrouter", api_mode="chat_completions", base_url="https://a.example/v1",
        api_key="")
    merge_preflight_compression_warning(
        result, agent=agent, messages=[{"role": "user", "content": "x"}])
    assert result.warning_message == ""
