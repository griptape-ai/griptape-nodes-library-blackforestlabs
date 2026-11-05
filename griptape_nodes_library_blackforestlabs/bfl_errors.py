"""Errors the Black Forest Labs nodes raise beyond what `NodeError` builds itself.

A missing or refused API key and a failed HTTP call come from `NodeError.from_missing_secret` and
`NodeError.from_request_exception`. This module covers a polling result that ends without an image.
"""

from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import NodeError

__all__ = ["API_KEY_ENV_VAR", "API_KEY_URL", "PROVIDER", "result_error"]

API_KEY_ENV_VAR = "BFL_API_KEY"
API_KEY_URL = "https://dashboard.bfl.ai/api/keys"
PROVIDER = "Black Forest Labs"

# Polling statuses for a request the API refused on content grounds.
MODERATED_STATUSES = ("Request Moderated", "Content Moderated")


def result_error(result: dict[str, Any]) -> NodeError:
    """The error for a polling result that ended without an image: moderated, errored, or failed."""
    status = result.get("status")
    if status in MODERATED_STATUSES:
        reasons = (result.get("details") or {}).get("Moderation Reasons") or ["Unknown"]
        message = (
            f"{PROVIDER} blocked the request: {', '.join(map(str, reasons))}. "
            "Raise 'Safety Tolerance' or reword the prompt."
        )
    elif reason := (result.get("result") or {}).get("error"):
        message = f"{PROVIDER} could not finish the request: {reason}"
    else:
        message = f"{PROVIDER} ended the request with status '{status}' and gave no reason."

    fields = {"status": status, "request_id": result.get("id")}
    return NodeError(message, fields={k: v for k, v in fields.items() if v}, response=result)
