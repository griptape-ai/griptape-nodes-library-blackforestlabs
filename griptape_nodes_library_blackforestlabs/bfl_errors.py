"""Errors the Black Forest Labs nodes raise beyond what `NodeError` builds itself.

A missing or refused API key and a failed HTTP call come from `NodeError.from_missing_secret` and
`NodeError.from_request_exception`. This module covers a polling result that ends without an image and
an API response that is missing what the node needs. Each carries what a user may need to quote to
support as `fields` and `response`.
"""

from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import NodeError

__all__ = [
    "API_KEY_ENV_VAR",
    "API_KEY_URL",
    "PROVIDER",
    "result_error",
    "unexpected_response_error",
]

API_KEY_ENV_VAR = "BFL_API_KEY"
API_KEY_URL = "https://dashboard.bfl.ai/api/keys"
PROVIDER = "Black Forest Labs"

# Polling statuses for a request the API refused on content grounds.
MODERATED_STATUSES = ("Request Moderated", "Content Moderated")


def result_error(result: dict[str, Any]) -> NodeError:
    """The error for a polling result that ended without an image: moderated, errored, or failed."""
    status = result.get("status")
    if status in MODERATED_STATUSES:
        message = (
            f"Black Forest Labs blocked the request: {', '.join(_moderation_reasons(result))}. "
            "Raise 'Safety Tolerance' or reword the prompt."
        )
    else:
        reason = _result_reason(result)
        message = (
            f"Black Forest Labs could not finish the request: {reason}"
            if reason
            else f"Black Forest Labs ended the request with status '{status}' and gave no reason."
        )
    return NodeError(message, fields=_result_fields(result), response=result)


def unexpected_response_error(message: str, result: Any) -> NodeError:
    """The error for an API response that is missing what the node needs, with the response attached."""
    body = result if isinstance(result, dict) and result else None
    return NodeError(message, fields=_result_fields(body) if body else None, response=body)


def _moderation_reasons(result: dict[str, Any]) -> list[str]:
    details = result.get("details")
    reasons = details.get("Moderation Reasons") if isinstance(details, dict) else None
    if isinstance(reasons, list) and reasons:
        return [str(reason) for reason in reasons]
    return ["Unknown"]


def _result_reason(result: dict[str, Any]) -> str | None:
    inner = result.get("result")
    reason = inner.get("error") if isinstance(inner, dict) else None
    return reason.strip() if isinstance(reason, str) and reason.strip() else None


def _result_fields(result: dict[str, Any] | None) -> dict[str, str | int | float | bool] | None:
    """The status and request ID of a polling result, for the user to quote to support."""
    if result is None:
        return None
    fields: dict[str, str | int | float | bool] = {}
    for key, name in (("status", "status"), ("id", "request_id")):
        value = result.get(key)
        if isinstance(value, str) and value:
            fields[name] = value
    return fields or None
