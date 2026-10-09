"""Regression tests for the single-query clarify guard (#94943).

``hermes chat -q`` wires the interactive prompt_toolkit clarify callback
unconditionally: a -q turn never builds the prompt_toolkit application, so
``CLIApp._clarify_callback`` polls its response queue with nothing able to
answer it — the turn hangs until ``agent.clarify_timeout`` expires (default
3600 s, 0 = unlimited). The gateway, cron jobs, the kanban dispatcher and
inter-agent wakeups all deliver work as ``hermes chat -q``, so an agent that
calls ``clarify`` in those turns stalls silently. The oneshot (-z) path
already answers immediately via ``_oneshot_clarify_callback``; this pins the
same headless behavior for -q, wired at the ``CLIAgentSetupMixin`` agent
construction site that already knows it is in single-query mode.
"""

from __future__ import annotations

def test_returns_immediate_undelivered_reply():
    from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback

    result = _single_query_clarify_callback([{
        "qid": "q0", "question": "Which timezone should I use?", "choices": None,
        "choices_offered": None, "multi_select": False}])
    assert result["answers"] == {}
    assert result["outcome"] == "undelivered"
    assert "no user available" in result["notice"]


class TestSingleQueryConsentGuard:
    """#107068: headless auto-decide must never pick authorization-semantic
    options -- a Recommended "authorize me" row is a silent self-authorization
    vector when no human is present to answer."""

    _REPRO = [
        "authorize me to compute these 9 rows directly (Recommended)",
        "configure A2A peer then hand off",
        "no aggregation, layout only",
    ]

    @staticmethod
    def _ask(choices):
        return [{"qid": "q0", "question": "How to proceed?", "choices": list(choices),
                 "choices_offered": list(choices), "multi_select": False}]

    def test_recommended_authorize_choice_is_declined_not_picked(self):
        from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback

        reply = _single_query_clarify_callback(self._ask(list(self._REPRO)))
        assert reply["answers"] == {}
        assert reply["outcome"] == "undelivered"
        notice = reply["notice"]
        assert "authorize me to compute these 9 rows directly" in notice  # explicitly named...
        assert "DECLINED" in notice  # ...as declined without a human...
        assert "do NOT pick them" in notice
        assert "Pick the best choices using your own judgment" not in notice  # ...never auto-picked

    def test_all_consent_choices_mean_stop_not_pick(self):
        from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback

        reply = _single_query_clarify_callback(
            self._ask(["authorize me to proceed (Recommended)", "allow me to continue"]))
        notice = reply["notice"]
        assert "DECLINED" in notice
        assert "Pick the best" not in notice
        assert "human input is needed" in notice

    def test_innocuous_choices_keep_exact_legacy_notice(self):
        """Non-consent choices must keep the byte-identical pre-#107068 notice."""
        from hermes_cli.cli_agent_setup_mixin import _single_query_clarify_callback

        reply = _single_query_clarify_callback(self._ask(["json", "yaml"]))
        assert reply["notice"] == (
            "single-query mode: no user available to answer. Pick the best choices "
            "using your own judgment, or make the most reasonable assumption you "
            "can, and continue."
        )

    def test_oneshot_callback_shares_the_guard(self):
        """The -z path has the same auto-pick shape; the sibling must not stay open."""
        from hermes_cli.oneshot import _oneshot_clarify_callback

        reply = _oneshot_clarify_callback(self._ask(list(self._REPRO)))
        assert "DECLINED" in reply["notice"]
        legacy = _oneshot_clarify_callback(self._ask(["json", "yaml"]))
        assert legacy["notice"] == (
            "oneshot mode: no user available. Pick the best choices using your own "
            "judgment, or make the most reasonable assumption you can, and continue."
        )
        assert "DECLINED" not in legacy["notice"]
