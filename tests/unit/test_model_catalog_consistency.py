"""Keep the `model_catalog` declarations in sync with the model lists the
library actually serves.

The catalog is the contract the library exposes to the platform (policy, key
support, UI grouping). Each node carries its served model list statically in
Python -- either as the `model_choices` a `ModelAccessComponent` decorates the
"model" parameter's dropdown with, a literal `Options(choices=[...])` trait
for nodes that don't gate the dropdown through the component, or (for
`FluxFill`) a single fixed provider model id with no dropdown at all. The
catalog and the served models must agree, and these tests are the guard that
keeps them from drifting.

The model lists are read directly from each node's source via `ast` rather
than importing the node modules. Importing pulls in the full `griptape_nodes`
engine package, which these tests don't otherwise need; parsing the source
keeps the check self-contained and immune to unrelated import-time failures.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]
LIBRARY_DIR = REPO_ROOT / "griptape_nodes_library_blackforestlabs"
LIBRARY_JSON = LIBRARY_DIR / "griptape_nodes_library.json"

# FluxFill has no "model" dropdown -- it always calls a single fixed BFL
# endpoint. The id is a literal baked into `_create_request` and
# `_create_image_artifact`, not a named constant.
FLUX_FILL_FIXED_MODEL = "flux-pro-1.0-fill"


def _load_library() -> dict[str, Any]:
    return json.loads(LIBRARY_JSON.read_text())


def _provider_model_id_by_catalog_id(library: dict[str, Any]) -> dict[str, str]:
    """Map every catalog model id to its provider_model_id, across all providers."""
    catalog = next(d for d in library["metadata"]["declarations"] if d["type"] == "model_catalog")
    return {
        model_id: model["provider_model_id"]
        for provider in catalog["providers"].values()
        for model_id, model in provider["models"].items()
    }


def _model_usage_ids(library: dict[str, Any], class_name: str) -> list[str]:
    node = next(n for n in library["nodes"] if n["class_name"] == class_name)
    usage = next(d for d in node["metadata"]["declarations"] if d["type"] == "model_usage")
    return usage["model_ids"]


def _nodes_with_model_usage(library: dict[str, Any]) -> list[str]:
    """Class names of every node that declares `model_usage`."""
    return [
        node["class_name"]
        for node in library["nodes"]
        if any(d.get("type") == "model_usage" for d in node.get("metadata", {}).get("declarations", []))
    ]


def _top_level_constants(tree: ast.Module) -> dict[str, Any]:
    """Map every top-level `NAME = <literal>` assignment to its literal value."""
    constants: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return constants


def _model_choices_from_source(file_path: Path, *, param_name: str = "model") -> list[str]:
    """Extract the model choices a node's model Parameter offers.

    Handles both ways a node can serve its dropdown list:

    - `ModelAccessComponent(parameter=..., model_choices=..., ...)`: the
      license-policy helper owns the `Options` trait, so the choices live in
      the component constructor's `model_choices` argument.
    - `Parameter(name=param_name, traits={Options(choices=...)})`: the plain
      trait, for nodes that don't gate the dropdown through the component.

    Either way, `choices`/`model_choices` may be written inline or as a
    reference to a module-level constant; both are resolved to a literal list.
    """
    tree = ast.parse(file_path.read_text())
    constants = _top_level_constants(tree)

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "ModelAccessComponent":
            continue
        choices_kwarg = next((kw for kw in node.keywords if kw.arg == "model_choices"), None)
        if choices_kwarg is None:
            continue
        if isinstance(choices_kwarg.value, ast.Name):
            return list(constants[choices_kwarg.value.id])
        return list(ast.literal_eval(choices_kwarg.value))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name_kwarg = next(
            (kw for kw in node.keywords if kw.arg == "name" and isinstance(kw.value, ast.Constant)),
            None,
        )
        if name_kwarg is None or name_kwarg.value.value != param_name:
            continue

        traits_kwarg = next((kw for kw in node.keywords if kw.arg == "traits"), None)
        if traits_kwarg is None:
            continue

        for options_call in ast.walk(traits_kwarg.value):
            if not (isinstance(options_call, ast.Call) and isinstance(options_call.func, ast.Name)):
                continue
            if options_call.func.id != "Options":
                continue
            choices_kwarg = next((kw for kw in options_call.keywords if kw.arg == "choices"), None)
            if choices_kwarg is None:
                continue
            if isinstance(choices_kwarg.value, ast.Name):
                return list(constants[choices_kwarg.value.id])
            return list(ast.literal_eval(choices_kwarg.value))

    msg = f"Could not find model choices for parameter '{param_name}' in {file_path}"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    ("class_name", "file_name"),
    [
        ("TextToImage", "text_to_image.py"),
        ("KontextImageEdit", "kontext_image_edit.py"),
        ("Flux2ImageGeneration", "flux_2_image_generation.py"),
    ],
)
def test_dropdown_node_model_usage_matches_source_choices(class_name: str, file_name: str) -> None:
    """The model dropdown in Python and the models the node declares must agree.

    Each node's `model_usage` ids resolve (through the catalog) to the same
    ordered provider model ids the node's `model` parameter actually offers.
    A mismatch means the manifest and the code drifted and one side needs
    updating.
    """
    library = _load_library()
    provider_model_id_by_catalog_id = _provider_model_id_by_catalog_id(library)

    declared = [provider_model_id_by_catalog_id[model_id] for model_id in _model_usage_ids(library, class_name)]
    served = _model_choices_from_source(LIBRARY_DIR / file_name)

    assert declared == served


def test_flux_fill_model_usage_matches_fixed_endpoint() -> None:
    """FluxFill has no dropdown -- it declares its single fixed BFL endpoint."""
    library = _load_library()
    provider_model_id_by_catalog_id = _provider_model_id_by_catalog_id(library)

    declared = [provider_model_id_by_catalog_id[model_id] for model_id in _model_usage_ids(library, "FluxFill")]

    assert declared == [FLUX_FILL_FIXED_MODEL]
    assert FLUX_FILL_FIXED_MODEL in (LIBRARY_DIR / "flux_fill.py").read_text()


def test_declared_models_resolve_uniquely_per_node() -> None:
    """Each node's declared models map one-to-one to provider model ids.

    Code that resolves a selected provider model id back to its stable
    catalog key (e.g. to declare a model invocation) matches within the
    node's own declared models. That match is unambiguous only when a node
    does not declare two catalog ids that share a `provider_model_id`; a
    duplicate would make the runtime resolver fail closed. Guard the
    manifest against introducing one.
    """
    library = _load_library()
    provider_model_id_by_catalog_id = _provider_model_id_by_catalog_id(library)

    duplicates: dict[str, list[str]] = {}
    for class_name in _nodes_with_model_usage(library):
        wire_ids = [provider_model_id_by_catalog_id[model_id] for model_id in _model_usage_ids(library, class_name)]
        repeated = sorted({wire_id for wire_id in wire_ids if wire_ids.count(wire_id) > 1})
        if repeated:
            duplicates[class_name] = repeated

    assert not duplicates, f"nodes declare catalog ids that share a provider_model_id: {duplicates}"
