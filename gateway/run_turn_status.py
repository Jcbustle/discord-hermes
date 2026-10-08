"""User-facing detail for the long-running gateway heartbeat."""

from collections.abc import Mapping
from typing import Any

from agent.display import build_status_phrase
from agent.session_activity import format_iteration_progress


def build_heartbeat_status_detail(activity: Mapping[str, Any] | None, *, include_iteration: bool) -> str:
    """Expose curated tool phrases; unknown identifiers leave the ordinary status intact."""
    if not activity:
        return ""
    parts = []
    if include_iteration:
        parts.append(format_iteration_progress(activity["api_call_count"], activity["max_iterations"]))
    action = activity.get("current_tool") or activity.get("last_activity_desc")
    if action:
        phrase = build_status_phrase(str(action), None)
        if phrase:
            parts.append(phrase)
    return " — " + ", ".join(parts) if parts else ""
