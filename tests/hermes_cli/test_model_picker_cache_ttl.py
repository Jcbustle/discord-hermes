"""Cache TTL legend alongside Cache pricing (#123945).

The legend must describe tiers this install can actually use — derived from
VALID_CACHE_TTLS / ``prompt_caching.cache_ttl`` and gated by the same disable
predicate as the runtime — not per-provider folklore ("OpenAI 5-10m auto"
matched no tier the repo can produce).
"""

import re

import pytest

from agent.agent_runtime_helpers import VALID_CACHE_TTLS
from hermes_cli import auth_model_picker

CACHE_PRICING = {
    "anthropic/claude-sonnet-4": {
        "prompt": "0.000003",
        "completion": "0.000015",
        "input_cache_read": "0.0000003",
    },
}


@pytest.fixture
def set_cache_ttl(monkeypatch):
    """Point the legend's config read at a fixed prompt_caching.cache_ttl."""

    def _set(value):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"prompt_caching": {"cache_ttl": value}},
        )

    return _set


def _legend_tiers(legend: str) -> set:
    return set(re.findall(r"\d+[mh]\b", legend))


def _rows(with_cache: bool = True):
    pricing = CACHE_PRICING if with_cache else {"model-a": {"prompt": "0.000003", "completion": "0.000015"}}
    return auth_model_picker._ModelPickerRows(
        list(pricing), pricing, current_model="", sale_chrome=False,
    )


def test_cache_ttl_legend_shown_with_cache_pricing(set_cache_ttl):
    set_cache_ttl("1h")
    title = _rows().menu_title()
    assert "Cache" in title
    assert "Cache TTL: 1h" in title


def test_no_ttl_legend_without_cache_pricing(set_cache_ttl):
    set_cache_ttl("1h")
    assert "TTL" not in _rows(with_cache=False).menu_title()


def test_legend_default_is_5m(set_cache_ttl):
    set_cache_ttl("5m")
    assert auth_model_picker.prompt_cache_ttl_legend() == "Cache TTL: 5m"


def test_legend_auto_names_this_sessions_tier_first(set_cache_ttl):
    set_cache_ttl("auto")
    legend = auth_model_picker.prompt_cache_ttl_legend()
    assert legend.startswith("Cache TTL: 1h")  # picker contexts are human-paced


@pytest.mark.parametrize("configured", ["5m", "1h", "auto", "2h"])
def test_legend_only_names_valid_cache_ttls(configured, set_cache_ttl):
    """Parity with the runtime's vocabulary — no foreign tiers like the old 5-10m."""
    set_cache_ttl(configured)
    legend = auth_model_picker.prompt_cache_ttl_legend()
    assert legend is not None
    assert _legend_tiers(legend) <= set(VALID_CACHE_TTLS), legend
    assert "OpenAI" not in legend and "Anthropic" not in legend and "Gemini" not in legend


def test_legend_absent_when_caching_disabled(set_cache_ttl):
    set_cache_ttl("off")
    assert auth_model_picker.prompt_cache_ttl_legend() is None
    title = _rows().menu_title()
    assert "Cache" in title  # the price column still shows
    assert "TTL" not in title


def test_unknown_value_keeps_runtime_default_5m(set_cache_ttl):
    set_cache_ttl("2h")
    assert auth_model_picker.prompt_cache_ttl_legend() == "Cache TTL: 5m"
