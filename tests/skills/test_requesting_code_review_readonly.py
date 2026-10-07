"""Regression for issue #106672.

The bundled `requesting-code-review` skill must stay read-only unless the
user explicitly authorizes mutation: a review/verify/inspect/report request
must not reach file edits, blanket staging, or a commit.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL = REPO_ROOT / "skills" / "software-development" / "requesting-code-review" / "SKILL.md"

# A routing sentence is gated when it carries one of these conditions.
_GATE_MARKERS = re.compile(r"only if|unless|gate|authoriz|explicitly requested|explicitly asked", re.IGNORECASE)
_ROUTE_VERB = re.compile(r"\b(?:proceed|go)\b", re.IGNORECASE)


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def _flattened() -> str:
    return " ".join(_text().split())


def _sentences() -> list[str]:
    return re.split(r"(?<=\.)\s+", _flattened())


def _routes_to(step: int) -> list[str]:
    """Sentences that direct the reader INTO `step` (proceed/go ... Step N)."""
    return [s for s in _sentences() if f"Step {step}" in s and _ROUTE_VERB.search(s)]


def test_no_blanket_stage():
    """Never stage the whole working tree; a bare `git add -A` captures unrelated changes."""
    assert "git add -A" not in _text()


def test_commit_stages_only_reviewed_paths():
    """Staging must stay scoped to the reviewed paths."""
    text = _flattened()
    assert "stage only the reviewed paths" in text
    assert "never the whole tree" in text


def test_authorization_gate_is_intact():
    """The gate section must spell out both conditions, not just its heading."""
    match = re.search(r"## Authorization gate[^\n]*\n(.*?)(?=\n## )", _text(), re.DOTALL)
    assert match, "authorization gate section is missing"
    gate = " ".join(match.group(0).split())
    assert "read-only" in gate
    assert "Do not proceed to Step 7 (auto-fix)" in gate
    assert "Do not proceed to Step 8 (commit)" in gate


def test_every_commit_route_is_gated():
    """Every route into Step 8 must carry its authorization condition (#106672 review)."""
    routes = _routes_to(8)
    assert len(routes) >= 3, f"expected the Step 8 routes, found: {routes!r}"
    for sentence in routes:
        assert _GATE_MARKERS.search(sentence), f"ungated route to Step 8: {sentence!r}"
    # The unconditional branch the issue reported must not come back.
    assert "**All passed:** Proceed to Step 8 (commit)." not in _text()


def test_every_auto_fix_route_is_gated():
    """Step 7 is entered only on an explicit fix request — heading and routes say so."""
    routes = _routes_to(7)
    assert routes, "no route into Step 7 found"
    for sentence in routes:
        assert _GATE_MARKERS.search(sentence), f"ungated route to Step 7: {sentence!r}"
    heading = next(line for line in _text().splitlines() if line.startswith("## Step 7"))
    assert "only when the user explicitly requested fixes" in heading


def test_review_only_default():
    """Review/verify/inspect requests are read-only by default."""
    assert "read-only by default" in _text()
