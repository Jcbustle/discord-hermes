"""Replay-echo retirement for interrupt placeholders (#132949, alongside #81841).

A hidden assistant row still reaches the provider as assistant content on replay.
Before the wording change it carried ``[response interrupted]`` — a short
natural-language phrase in the model's own prior turn, which the model reproduces
verbatim (clean ``finish_reason=stop`` answering an ordinary instruction with just
the placeholder). The prep filter neutralises (never drops) hidden rows carrying a
pre-change spelling on a copy: removal could form ``tool -> user`` (#48879) or a
``user -> user`` pair that repair merges.
"""

LEGACY = "[response interrupted]"
NEW_PLACEHOLDER = "[interrupt: no assistant output for this turn]"


def _agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from run_agent import AIAgent
    from hermes_state import SessionDB
    return AIAgent(session_db=SessionDB(db_path=tmp_path / "proof.db"),
                   model="test-model", provider="openai-compat", api_key="test",
                   base_url="http://127.0.0.1:1/v1", max_iterations=4,
                   quiet_mode=True, skip_context_files=True, skip_memory=True)


def _prepare(tmp_path, monkeypatch, messages):
    """Run prepare_iteration and return the messages it produced."""
    from agent.turn_context import _reset_per_turn_agent_state
    from agent.turn_iteration_prep import prepare_iteration

    agent = _agent(tmp_path, monkeypatch)
    try:
        _reset_per_turn_agent_state(agent)
        user_message = messages[-1].get("content") if messages else None
        prep = prepare_iteration(
            agent, messages=messages, api_call_count=1,
            user_message=user_message, current_turn_user_idx=max(len(messages) - 1, 0),
        )
        return list(prep.messages)
    finally:
        agent._session_db.close()


def _hidden_row(text):
    return {"role": "assistant", "content": "", "display_kind": "hidden", "api_content": text}


def test_tool_tail_row_is_kept_and_neutralised(tmp_path, monkeypatch):
    """assistant(tool_calls) -> tool -> hidden(legacy) -> user: dropping would
    recreate tool -> user (#48879). The row stays, but its echoable text is
    replaced with the post-fix placeholder. The persisted redirect shape
    user -> hidden(legacy) -> user(checkpoint) must not collapse either: a
    dropped row leaves user -> user, which repair merges, losing the
    correction's checkpoint sidecar and rewriting the first row in place."""
    import copy
    checkpoint = "[Context from the interrupted assistant response]\nhalf a poem\n\nmake it short"
    redirect = [
        {"role": "user", "content": "write a poem"},
        _hidden_row(LEGACY),
        {"role": "user", "content": "make it short", "api_content": checkpoint},
    ]
    stored = copy.deepcopy(redirect)
    (tmp_path / "redirect").mkdir()
    out = _prepare(tmp_path / "redirect", monkeypatch, redirect)
    assert redirect == stored, "stored rows were rewritten in place"
    assert [m["role"] for m in out] == ["user", "assistant", "user"], out
    assert out[1]["api_content"] == NEW_PLACEHOLDER
    assert out[2].get("api_content") == checkpoint, f"checkpoint sidecar lost: {out}"

    messages = [
        {"role": "user", "content": "edit the file"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "patch", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok edited"},
        _hidden_row(LEGACY),
        {"role": "user", "content": "they do! increase the timing"},
    ]
    out = _prepare(tmp_path, monkeypatch, messages)
    assert messages[3]["api_content"] == LEGACY  # durable row dict is never rewritten in place

    hidden = [m for m in out
              if m.get("role") == "assistant" and m.get("display_kind") == "hidden"]
    assert len(hidden) == 1, f"tool-tail row was dropped (tool -> user would revive): {out}"
    row = hidden[0]
    assert row.get("content") == ""
    assert row.get("api_content") == NEW_PLACEHOLDER, (
        f"legacy text survived neutralisation: {row.get('api_content')!r}"
    )
    # The sequence still has no tool -> user adjacency.
    for i in range(len(out) - 1):
        if out[i].get("role") == "tool":
            assert out[i + 1].get("role") != "user", (
                f"role-alternation violation: tool -> user at index {i}: {out}"
            )


def test_visible_echoed_reply_is_neutralised_but_tool_call_rows_are_not(tmp_path, monkeypatch):
    """A reply the model already produced as just the legacy placeholder is a visible row; replayed, it
    seeds the same echo. It is neutralised on the copy like a hidden row; a tool-call row never is."""
    import copy
    tool_row = {"role": "assistant", "content": LEGACY, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "patch", "arguments": "{}"}}]}
    messages = [
        {"role": "user", "content": "summarise the log"},
        {"role": "assistant", "content": LEGACY},
        {"role": "user", "content": "now fix it"},
        tool_row,
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "user", "content": "thanks"},
    ]
    stored = copy.deepcopy(messages)
    out = _prepare(tmp_path, monkeypatch, messages)
    assert messages == stored, "stored rows were rewritten in place"
    echoed = out[1]
    assert echoed["role"] == "assistant" and echoed.get("content") == ""
    assert echoed.get("display_kind") == "hidden"  # not an empty visible bubble
    assert echoed.get("api_content") == NEW_PLACEHOLDER
    assert any(m.get("tool_calls") and m.get("content") == LEGACY for m in out), out


def test_placeholder_before_a_reply_is_dropped_and_exact_items_retired(tmp_path, monkeypatch):
    """A placeholder followed by another assistant row is dropped (repair would fold the reply into the
    hidden row and hide it, text or tool calls alike); a kept one loses its exact Responses items."""
    call = {"id": "c1", "type": "function", "function": {"name": "patch", "arguments": "{}"}}
    for follower in ({"role": "assistant", "content": "real answer"},
                     {"role": "assistant", "content": "", "tool_calls": [call]}):
        messages = [{"role": "user", "content": "hi"}, _hidden_row(LEGACY), follower]
        if follower.get("tool_calls"):
            messages.append({"role": "tool", "tool_call_id": "c1", "content": "ok"})
        messages.append({"role": "user", "content": "continue"})
        out = _prepare(tmp_path, monkeypatch, messages)
        assistants = [m for m in out if m.get("role") == "assistant"]
        assert len(assistants) == 1 and assistants[0].get("display_kind") != "hidden", out

    # The dropped durable placeholder is retired onto the reply that replaces it (not left unnamed for an
    # in-place compaction to re-sequence), and the stored reply dict itself is untouched.
    reply = {"role": "assistant", "content": "real answer", "_row_id": 12, "_absorbed_row_ids": []}
    stored = [{"role": "user", "content": "hi", "_row_id": 10}, {**_hidden_row(LEGACY), "_row_id": 11}, reply,
              {"role": "user", "content": "continue", "_row_id": 13}]
    out = _prepare(tmp_path, monkeypatch, stored)
    survivor = next(m for m in out if m.get("content") == "real answer")
    assert 11 in survivor.get("_absorbed_row_ids", []) and reply["_absorbed_row_ids"] == [], out

    # A thinking-only row then a real reply: the run reaches the wire, so the placeholder is dropped and
    # the reply stays visible (kept, repair folds the whole run into the hidden row).
    out = _prepare(tmp_path, monkeypatch, [{"role": "user", "content": "hi"}, _hidden_row(LEGACY),
                                           {"role": "assistant", "content": "", "reasoning_content": "hmm"},
                                           {"role": "assistant", "content": "real answer"},
                                           {"role": "user", "content": "continue"}])
    assert not any(m.get("display_kind") == "hidden" for m in out), out

    # A thinking-only follower never reaches the wire, so the placeholder stays as the tool tail's closer.
    thinking = {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}], "reasoning_content": "hmm"}
    out = _prepare(tmp_path, monkeypatch, [{"role": "user", "content": "hi"}, _hidden_row(LEGACY), thinking,
                                           {"role": "user", "content": "continue"}])
    assert any(m.get("display_kind") == "hidden" and m.get("role") == "assistant" for m in out), out

    # A kept id-less durable echo records its loaded fields before its text changes (compaction coverage).
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from agent.conversation_compression_archive import OWN_ROW, RETIRED_DURABLE_ROWS
    durable = {"role": "assistant", "content": LEGACY, _DB_PERSISTED_MARKER: True, RETIRED_DURABLE_ROWS: []}
    out = _prepare(tmp_path, monkeypatch, [{"role": "user", "content": "a"}, durable, {"role": "user", "content": "b"}])
    own = [r for r in out[1].get(RETIRED_DURABLE_ROWS, []) if r.get(OWN_ROW)]
    assert own and own[0].get("content") == LEGACY and durable[RETIRED_DURABLE_ROWS] == [], out

    echoed = {"role": "assistant", "content": LEGACY, "bedrock_content_blocks": [{"text": LEGACY}],
              "codex_message_items": [{"type": "message", "content": [{"type": "output_text", "text": LEGACY}]}]}
    out = _prepare(tmp_path, monkeypatch, [{"role": "user", "content": "a"}, echoed, {"role": "user", "content": "b"}])
    assert "codex_message_items" not in out[1] and LEGACY not in repr(out), out
