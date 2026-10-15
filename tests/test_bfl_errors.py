"""Tests for the errors the Black Forest Labs nodes raise."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from bfl_errors import (
    missing_api_key_error,
    request_error,
    result_error,
    unexpected_response_error,
)
from griptape_nodes.exe_types.core_types import NodeError

NODES = [
    ("flux_fill", "FluxFill"),
    ("kontext_image_edit", "KontextImageEdit"),
    ("flux_2_image_generation", "Flux2ImageGeneration"),
    ("text_to_image", "TextToImage"),
]


def _http_error(status: int, *, body: Any = None, text: str = "") -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = (json.dumps(body) if body is not None else text).encode()
    return requests.HTTPError(f"{status} Error", response=response)


def test_missing_api_key_links_to_the_secret_and_the_key_page() -> None:
    error = missing_api_key_error()

    assert str(error) == "BFL_API_KEY is not set. Add it in Settings → API Keys & Secrets, then run the node again."
    assert [link.url for link in error.links] == [
        "#settings-secrets?filter=BFL_API_KEY",
        "https://dashboard.bfl.ai/api/keys",
    ]


def test_request_error_uses_the_apis_reason_and_attaches_the_response() -> None:
    body = {"detail": "Image is too large."}

    error = request_error(_http_error(400, body=body))

    assert str(error) == "Image is too large."
    assert error.fields == {"status_code": 400}
    assert error.response == body
    assert error.links == []


def test_request_error_lists_each_validation_problem() -> None:
    body = {
        "detail": [
            {"loc": ["body", "width"], "msg": "Input should be a multiple of 32", "type": "value_error"},
            {"loc": ["body", "height"], "msg": "Input should be at most 2048", "type": "value_error"},
        ]
    }

    error = request_error(_http_error(422, body=body))

    assert str(error) == "width: Input should be a multiple of 32; height: Input should be at most 2048"
    assert error.fields == {"status_code": 422}


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (402, "Black Forest Labs says payment is required. Check your account's credits."),
        (429, "Black Forest Labs is rate-limiting requests. Wait a moment and run the node again."),
        (502, "Black Forest Labs returned an error (HTTP 502)."),
    ],
)
def test_request_error_without_a_reason_describes_the_status(status: int, expected: str) -> None:
    error = request_error(_http_error(status, text="<html>Bad gateway</html>"))

    assert str(error) == expected
    assert error.fields == {"status_code": status}
    assert error.response is None


@pytest.mark.parametrize(
    ("status", "detail"),
    [(403, "Not authenticated"), (401, "Not authenticated"), (422, "Invalid API key format")],
)
def test_a_refused_key_links_to_the_secret(status: int, detail: str) -> None:
    """BFL answers a well-formed unknown key with 403, and a malformed key with 422."""
    error = request_error(_http_error(status, body={"detail": detail}))

    assert str(error) == detail
    assert [(link.label, link.url) for link in error.links] == [
        ("Update the API key", "#settings-secrets?filter=BFL_API_KEY"),
        ("Get a Black Forest Labs API key", "https://dashboard.bfl.ai/api/keys"),
    ]


def test_a_validation_problem_that_is_not_about_the_key_has_no_key_links() -> None:
    body = {"detail": [{"loc": ["body", "width"], "msg": "Input should be a multiple of 32"}]}

    assert request_error(_http_error(422, body=body)).links == []


def test_request_error_without_a_response_says_the_api_could_not_be_reached() -> None:
    error = request_error(requests.ConnectionError("no route to host"))

    assert str(error) == "Could not reach Black Forest Labs: no route to host"
    assert error.fields == {}


def test_a_moderated_result_names_the_reasons_and_the_setting_to_change() -> None:
    result = {"id": "req-1", "status": "Request Moderated", "details": {"Moderation Reasons": ["Violence", "Weapons"]}}

    error = result_error(result)

    assert str(error) == (
        "Black Forest Labs blocked the request: Violence, Weapons. Raise 'Safety Tolerance' or reword the prompt."
    )
    assert error.fields == {"status": "Request Moderated", "request_id": "req-1"}
    assert error.response == result


def test_a_moderated_result_without_reasons_says_unknown() -> None:
    assert "blocked the request: Unknown." in str(result_error({"status": "Content Moderated"}))


def test_a_failed_result_uses_the_reason_it_gave() -> None:
    result = {"id": "req-2", "status": "Error", "result": {"error": "The input image could not be decoded."}}

    error = result_error(result)

    assert str(error) == "Black Forest Labs could not finish the request: The input image could not be decoded."
    assert error.fields == {"status": "Error", "request_id": "req-2"}
    assert error.response == result


def test_a_failed_result_without_a_reason_says_so() -> None:
    error = result_error({"status": "Failed"})

    assert str(error) == "Black Forest Labs ended the request with status 'Failed' and gave no reason."


def test_unexpected_response_error_attaches_the_response() -> None:
    result = {"id": "req-3", "unexpected": True}

    error = unexpected_response_error("Black Forest Labs did not return a polling URL for the request.", result)

    assert error.fields == {"request_id": "req-3"}
    assert error.response == result


def test_unexpected_response_error_with_nothing_to_attach() -> None:
    error = unexpected_response_error("No image.", None)

    assert error.fields == {}
    assert error.response is None


@pytest.mark.parametrize(("module_name", "class_name"), NODES)
def test_validation_reports_each_problem_without_the_node_name(module_name: str, class_name: str) -> None:
    module = importlib.import_module(module_name)
    node = getattr(module, class_name)(name="Some Node")

    with patch.object(module, "GriptapeNodes") as engine:
        engine.SecretsManager.return_value.get_secret.return_value = None
        errors = node.validate_before_node_run()

    assert errors
    assert all(isinstance(error, NodeError) for error in errors)
    assert all("Some Node" not in str(error) for error in errors)
    key_errors = [error for error in errors if "BFL_API_KEY" in str(error)]
    assert len(key_errors) == 1
    assert key_errors[0].links[0].url == "#settings-secrets?filter=BFL_API_KEY"


def _load(module_name: str, class_name: str) -> tuple[Any, Any]:
    module = importlib.import_module(module_name)
    return module, getattr(module, class_name)(name="Some Node")


def _poll(module: Any, node: Any, result: dict[str, Any]) -> None:
    """Run the node's polling loop against one canned API result."""
    response = MagicMock(status_code=200)
    response.json.return_value = result
    with patch.object(module.requests, "get", return_value=response), patch.object(module.time, "sleep"):
        if hasattr(node, "_poll_for_result"):
            node._poll_for_result("key", "https://api.example.com/poll")
        else:
            node._poll_and_process_result("key", "https://api.example.com/poll", "png")


@pytest.mark.parametrize(("module_name", "class_name"), NODES)
def test_an_api_rejection_when_creating_the_request_keeps_its_status_and_response(
    module_name: str, class_name: str
) -> None:
    module, node = _load(module_name, class_name)
    body = {"detail": [{"loc": ["body", "width"], "msg": "Input should be a multiple of 32"}]}
    response = MagicMock()
    response.raise_for_status.side_effect = _http_error(422, body=body)

    with patch.object(module.requests, "post", return_value=response), pytest.raises(NodeError) as raised:
        node._create_request("key", {})

    assert str(raised.value) == "width: Input should be a multiple of 32"
    assert raised.value.fields == {"status_code": 422}
    assert raised.value.response == body


@pytest.mark.parametrize(("module_name", "class_name"), NODES)
def test_a_response_without_a_polling_url_is_attached(module_name: str, class_name: str) -> None:
    module, node = _load(module_name, class_name)
    response = MagicMock()
    response.json.return_value = {"id": "req-9"}

    with patch.object(module.requests, "post", return_value=response), pytest.raises(NodeError) as raised:
        node._create_request("key", {})

    assert str(raised.value) == "Black Forest Labs did not return a polling URL for the request."
    assert raised.value.response == {"id": "req-9"}


@pytest.mark.parametrize(("module_name", "class_name"), [n for n in NODES if n[0] != "text_to_image"])
def test_a_moderated_request_fails_the_node_with_the_reasons(module_name: str, class_name: str) -> None:
    module, node = _load(module_name, class_name)
    result = {"id": "req-4", "status": "Request Moderated", "details": {"Moderation Reasons": ["Violence"]}}

    with pytest.raises(NodeError) as raised:
        _poll(module, node, result)

    assert "blocked the request: Violence." in str(raised.value)
    assert raised.value.fields == {"status": "Request Moderated", "request_id": "req-4"}
    assert raised.value.response == result


@pytest.mark.parametrize(("module_name", "class_name"), NODES)
def test_a_failed_request_fails_the_node_with_its_reason(module_name: str, class_name: str) -> None:
    module, node = _load(module_name, class_name)
    result = {"id": "req-5", "status": "Error", "result": {"error": "The input image could not be decoded."}}

    with pytest.raises(NodeError) as raised:
        _poll(module, node, result)

    assert str(raised.value) == "Black Forest Labs could not finish the request: The input image could not be decoded."
    assert raised.value.response == result


LIBRARY_DIR = Path(__file__).resolve().parent.parent / "griptape_nodes_library_blackforestlabs"


@pytest.mark.parametrize(("module_name", "class_name"), NODES)
def test_a_node_file_loads_the_way_the_engine_loads_it(
    module_name: str, class_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine loads each node file on its own, with the library folder on sys.path.

    There is no parent package, and the library's package name is not importable there, so a node
    has to import its sibling modules by bare name. The dev environment installs the project, which
    would hide a package-name import, so make the package unimportable here.
    """
    monkeypatch.syspath_prepend(str(LIBRARY_DIR))
    monkeypatch.setitem(sys.modules, "griptape_nodes_library_blackforestlabs", None)

    spec = importlib.util.spec_from_file_location(f"engine_loaded_{module_name}", LIBRARY_DIR / f"{module_name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, class_name)


@pytest.mark.parametrize(("module_name", "class_name"), NODES)
def test_nothing_is_checked_before_the_flow_starts(module_name: str, class_name: str) -> None:
    """A problem found before the flow starts shows as a toast with no link and no error on a node's header.

    The key and the inputs are checked when the node runs instead.
    """
    module, node = _load(module_name, class_name)

    with patch.object(module, "GriptapeNodes") as engine:
        engine.SecretsManager.return_value.get_secret.return_value = None
        before_flow = node.validate_before_workflow_run()

    assert not before_flow


@pytest.mark.parametrize(("module_name", "class_name"), NODES)
def test_a_missing_key_is_reported_when_the_node_runs_with_a_link_to_the_secret(
    module_name: str, class_name: str
) -> None:
    module, node = _load(module_name, class_name)

    with patch.object(module, "GriptapeNodes") as engine:
        engine.SecretsManager.return_value.get_secret.return_value = None
        errors = node.validate_before_node_run()

    key_errors = [error for error in errors or [] if "BFL_API_KEY" in str(error)]
    assert len(key_errors) == 1
    assert key_errors[0].links[0].url == "#settings-secrets?filter=BFL_API_KEY"


@pytest.mark.parametrize(("module_name", "class_name"), NODES)
def test_missing_inputs_are_reported_when_the_node_runs(module_name: str, class_name: str) -> None:
    """Inputs can come from upstream nodes, so they are checked when the node runs, where the header shows them."""
    module, node = _load(module_name, class_name)

    with patch.object(module, "GriptapeNodes") as engine:
        engine.SecretsManager.return_value.get_secret.return_value = "key"
        when_run = node.validate_before_node_run()

    assert when_run
    assert all(isinstance(error, NodeError) for error in when_run)


@pytest.mark.parametrize(("module_name", "class_name"), NODES)
def test_a_malformed_key_when_creating_the_request_links_to_the_secret(module_name: str, class_name: str) -> None:
    module, node = _load(module_name, class_name)
    response = MagicMock()
    response.raise_for_status.side_effect = _http_error(422, body={"detail": "Invalid API key format"})

    with patch.object(module.requests, "post", return_value=response), pytest.raises(NodeError) as raised:
        node._create_request("key", {})

    assert str(raised.value) == "Invalid API key format"
    assert raised.value.fields == {"status_code": 422}
    assert raised.value.links[0].url == "#settings-secrets?filter=BFL_API_KEY"
