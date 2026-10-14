"""Errors the Black Forest Labs nodes raise.

The four nodes fail in the same ways: the API key is missing, the API rejects a request, or a result
comes back moderated, failed, or without an image. Each error carries what a user may need to quote to
support, or to fix the problem, as `fields`, `response`, and `links`.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests
from griptape_nodes.exe_types.core_types import NodeError, NodeErrorLink

__all__ = [
    "API_KEY_ENV_VAR",
    "missing_api_key_error",
    "request_error",
    "result_error",
    "unexpected_response_error",
]

API_KEY_ENV_VAR = "BFL_API_KEY"
API_KEY_URL = "https://dashboard.bfl.ai/api/keys"

# Polling statuses for a request the API refused on content grounds.
MODERATED_STATUSES = ("Request Moderated", "Content Moderated")

# What to say for an HTTP status when the API's response has no reason of its own.
_HTTP_STATUS_REASONS = {
    401: f"Black Forest Labs did not accept the API key. Check {API_KEY_ENV_VAR} in Settings.",
    402: "Black Forest Labs says payment is required. Check your account's credits.",
    403: f"Black Forest Labs did not accept the API key. Check {API_KEY_ENV_VAR} in Settings.",
    429: "Black Forest Labs is rate-limiting requests. Wait a moment and run the node again.",
}
_KEY_REJECTED_STATUSES = (401, 403)


def _key_links() -> list[NodeErrorLink]:
    return [
        NodeErrorLink(label="Add the API key", url=f"#settings-secrets?filter={quote(API_KEY_ENV_VAR)}"),
        NodeErrorLink(label="Get a Black Forest Labs API key", url=API_KEY_URL),
    ]


def missing_api_key_error() -> NodeError:
    """The error for a run with no Black Forest Labs API key, linking to the secret in Settings."""
    return NodeError(
        f"{API_KEY_ENV_VAR} is not set. Add it in Settings → API Keys & Secrets, then run the node again.",
        links=_key_links(),
    )


def request_error(error: requests.RequestException) -> NodeError:
    """The error for a failed call to the API, with the status code and the API's response attached."""
    response = error.response
    if response is None:
        return NodeError(f"Could not reach Black Forest Labs: {error}")

    body = _json_body(response)
    reason = _api_reason(body) or _HTTP_STATUS_REASONS.get(
        response.status_code, f"Black Forest Labs returned an error (HTTP {response.status_code})."
    )
    links = _key_links() if response.status_code in _KEY_REJECTED_STATUSES else None
    return NodeError(reason, fields={"status_code": response.status_code}, response=body, links=links)


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


def _json_body(response: requests.Response) -> dict[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) and body else None


def _api_reason(body: dict[str, Any] | None) -> str | None:
    """The API's own explanation: a `detail` string, a list of validation problems, or a message."""
    if body is None:
        return None
    detail = body.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    if isinstance(detail, list):
        problems = [_validation_problem(item) for item in detail if isinstance(item, dict)]
        problems = [problem for problem in problems if problem]
        if problems:
            return "; ".join(problems)
    for key in ("message", "error"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _validation_problem(item: dict[str, Any]) -> str | None:
    """One entry of a validation `detail` list, such as `width: Input should be a multiple of 32`."""
    message = item.get("msg")
    if not isinstance(message, str) or not message.strip():
        return None
    location = item.get("loc")
    if isinstance(location, list) and location and isinstance(location[-1], str):
        return f"{location[-1]}: {message.strip()}"
    return message.strip()


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
