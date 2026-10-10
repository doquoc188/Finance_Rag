"""Offline contracts for the optional Microsoft GraphRAG vertical slice."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from finrag.graph_rag import (
    GraphQueryMode,
    GraphQueryResult,
    GraphRAGArtifactError,
    GraphRAGConfigurationError,
    GraphRAGDependencyError,
    GraphRAGQueryError,
    MicrosoftGraphRAGBackend,
    load_graph_artifacts,
    microsoft,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "evals" / "graphrag" / "fixtures" / "curated_finance"
WORKSPACE = PROJECT_ROOT / "evals" / "graphrag" / "workspace"
FAKE_KEY = "test-not-a-real-key"


class FakeFrame:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def to_dict(self, *, orient: str) -> list[dict[str, object]]:
        assert orient == "records"
        return self.rows


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "fixture"
    shutil.copytree(FIXTURE_DIR, target)
    return target


def _replace_json(root: Path, name: str, value: object) -> None:
    (root / name).write_text(json.dumps(value), encoding="utf-8")


def _read_json(root: Path, name: str) -> Any:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _bindings(
    *,
    local_search: Any,
    global_search: Any,
    config_calls: list[tuple[Path, dict[str, object]]],
) -> SimpleNamespace:
    def load_config(root: Path, *, cli_overrides: dict[str, object]) -> object:
        config_calls.append((root, cli_overrides))
        return object()

    return SimpleNamespace(
        dataframe=FakeFrame,
        load_config=load_config,
        local_search=local_search,
        global_search=global_search,
    )


def test_generic_import_does_not_require_graphrag_or_pandas() -> None:
    code = """
import importlib.abc
import sys

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'graphrag' or fullname.startswith('graphrag.') or fullname == 'pandas':
            raise AssertionError(f'optional import attempted: {fullname}')
        return None

sys.meta_path.insert(0, BlockOptional())
import finrag
import finrag.config
import finrag.graph_rag
print('imports-ok')
"""
    environment = dict(os.environ)
    environment.pop("GRAPHRAG_API_KEY", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "imports-ok"


def test_curated_fixture_counts_and_cross_document_chain() -> None:
    artifacts = load_graph_artifacts(FIXTURE_DIR)

    assert len(artifacts.documents) == 4
    assert len(artifacts.entities) == 5
    assert len(artifacts.relationships) == 4
    assert len(artifacts.text_units) == 4
    assert len(artifacts.communities) == 1
    assert len(artifacts.community_reports) == 1

    relationships = {item.id: item for item in artifacts.relationships}
    chain = [
        relationships["rel_northstar_expansion"],
        relationships["rel_expansion_delta"],
        relationships["rel_delta_harbor"],
    ]
    assert chain[0].target == chain[1].source
    assert chain[1].target == chain[2].source
    assert [artifacts.document_ids_for_text_units(item.text_unit_ids) for item in chain] == [
        ("doc_northstar_2024",),
        ("doc_delta_2024",),
        ("doc_harbor_2024",),
    ]


@pytest.mark.parametrize(
    ("filename", "replacement", "mode", "message"),
    [
        ("entities.json", [], GraphQueryMode.LOCAL, "entities must not be empty"),
        ("relationships.json", [], GraphQueryMode.LOCAL, "relationships must not be empty"),
        ("text_units.json", [], GraphQueryMode.LOCAL, "require text units"),
        ("community_reports.json", [], GraphQueryMode.GLOBAL, "require community reports"),
    ],
)
def test_required_artifacts_fail_explicitly(
    tmp_path: Path,
    filename: str,
    replacement: object,
    mode: GraphQueryMode,
    message: str,
) -> None:
    fixture = _copy_fixture(tmp_path)
    _replace_json(fixture, filename, replacement)

    with pytest.raises(GraphRAGArtifactError, match=message):
        load_graph_artifacts(fixture, mode=mode)


def test_duplicate_entity_ids_fail_explicitly(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    entities = _read_json(fixture, "entities.json")
    entities.append(dict(entities[0]))
    _replace_json(fixture, "entities.json", entities)

    with pytest.raises(GraphRAGArtifactError, match="Duplicate entity ID"):
        load_graph_artifacts(fixture)


def test_unknown_relationship_endpoint_fails_explicitly(tmp_path: Path) -> None:
    fixture = _copy_fixture(tmp_path)
    relationships = _read_json(fixture, "relationships.json")
    relationships[0]["target"] = "ent_missing"
    _replace_json(fixture, "relationships.json", relationships)

    with pytest.raises(GraphRAGArtifactError, match="unknown entity IDs"):
        load_graph_artifacts(fixture)


def test_local_dispatches_to_microsoft_boundary_and_translates_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_calls: list[dict[str, object]] = []
    config_calls: list[tuple[Path, dict[str, object]]] = []

    async def local_search(**kwargs: object) -> tuple[str, object]:
        local_calls.append(kwargs)
        return (
            "Deterministic local answer",
            {
                "entities": FakeFrame([{"id": "ent_expansion"}]),
                "relationships": FakeFrame([{"id": "rel_expansion_delta"}]),
            },
        )

    async def global_search(**kwargs: object) -> tuple[str, object]:
        raise AssertionError(f"GLOBAL must not be called: {sorted(kwargs)}")

    monkeypatch.setattr(
        microsoft,
        "_load_microsoft_bindings",
        lambda: _bindings(
            local_search=local_search,
            global_search=global_search,
            config_calls=config_calls,
        ),
    )
    backend = MicrosoftGraphRAGBackend(
        root_dir=WORKSPACE,
        artifact_dir=FIXTURE_DIR,
        environment={"GRAPHRAG_API_KEY": FAKE_KEY},
    )

    result = asyncio.run(
        backend.query("Which supplier supports the program?", mode=GraphQueryMode.LOCAL)
    )

    assert isinstance(result, GraphQueryResult)
    assert result.answer == "Deterministic local answer"
    assert result.mode is GraphQueryMode.LOCAL
    assert result.provenance[0].document_ids == (
        "doc_northstar_2024",
        "doc_delta_2024",
        "doc_market_2024",
    )
    assert result.provenance[1].document_ids == ("doc_delta_2024",)
    assert len(local_calls) == 1
    assert local_calls[0]["query"] == "Which supplier supports the program?"
    assert local_calls[0]["covariates"] is None
    assert local_calls[0]["community_level"] == 0
    assert isinstance(local_calls[0]["entities"], FakeFrame)
    assert len(config_calls) == 1
    assert config_calls[0][0] == WORKSPACE


def test_global_dispatches_to_microsoft_boundary_and_translates_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_calls: list[dict[str, object]] = []
    config_calls: list[tuple[Path, dict[str, object]]] = []

    async def local_search(**kwargs: object) -> tuple[str, object]:
        raise AssertionError(f"LOCAL must not be called: {sorted(kwargs)}")

    async def global_search(**kwargs: object) -> tuple[str, object]:
        global_calls.append(kwargs)
        return (
            "Deterministic global answer",
            {"reports": FakeFrame([{"id": "report_fixture_finance"}])},
        )

    monkeypatch.setattr(
        microsoft,
        "_load_microsoft_bindings",
        lambda: _bindings(
            local_search=local_search,
            global_search=global_search,
            config_calls=config_calls,
        ),
    )
    backend = MicrosoftGraphRAGBackend(
        root_dir=WORKSPACE,
        artifact_dir=FIXTURE_DIR,
        environment={"GRAPHRAG_API_KEY": FAKE_KEY},
    )

    result = asyncio.run(
        backend.query("Summarize the dependency chain.", mode=GraphQueryMode.GLOBAL)
    )

    assert result.answer == "Deterministic global answer"
    assert result.mode is GraphQueryMode.GLOBAL
    assert result.provenance[0].document_ids == (
        "doc_northstar_2024",
        "doc_delta_2024",
        "doc_harbor_2024",
        "doc_market_2024",
    )
    assert len(global_calls) == 1
    assert global_calls[0]["dynamic_community_selection"] is False
    assert global_calls[0]["query"] == "Summarize the dependency chain."


def test_missing_key_fails_before_optional_import_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_load() -> None:
        raise AssertionError("optional dependency boundary was reached")

    monkeypatch.setattr(microsoft, "_load_microsoft_bindings", forbidden_load)
    backend = MicrosoftGraphRAGBackend(
        root_dir=WORKSPACE,
        artifact_dir=FIXTURE_DIR,
        environment={},
    )

    with pytest.raises(GraphRAGConfigurationError, match="GRAPHRAG_API_KEY is required"):
        asyncio.run(backend.query("A live question", mode=GraphQueryMode.GLOBAL))


def test_dependency_absence_fails_only_on_concrete_adapter_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency() -> None:
        raise ModuleNotFoundError("No module named 'graphrag'")

    monkeypatch.setattr(microsoft, "_load_microsoft_bindings", missing_dependency)
    backend = MicrosoftGraphRAGBackend(
        root_dir=WORKSPACE,
        artifact_dir=FIXTURE_DIR,
        environment={"GRAPHRAG_API_KEY": FAKE_KEY},
    )

    with pytest.raises(GraphRAGDependencyError, match="graphrag extra"):
        asyncio.run(backend.query("A live question", mode=GraphQueryMode.GLOBAL))


def test_workspace_dotenv_support_uses_only_the_fake_fixture_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "settings.yaml").write_text("test: true\n", encoding="utf-8")
    (workspace / ".env").write_text(f"GRAPHRAG_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.delenv("GRAPHRAG_API_KEY", raising=False)
    config_calls: list[tuple[Path, dict[str, object]]] = []

    async def global_search(**kwargs: object) -> tuple[str, object]:
        return "Workspace env accepted", {}

    async def local_search(**kwargs: object) -> tuple[str, object]:
        raise AssertionError(f"LOCAL must not be called: {sorted(kwargs)}")

    monkeypatch.setattr(
        microsoft,
        "_load_microsoft_bindings",
        lambda: _bindings(
            local_search=local_search,
            global_search=global_search,
            config_calls=config_calls,
        ),
    )
    backend = MicrosoftGraphRAGBackend(root_dir=workspace, artifact_dir=FIXTURE_DIR)

    result = asyncio.run(backend.query("A live question", mode=GraphQueryMode.GLOBAL))

    assert result.answer == "Workspace env accepted"
    assert FAKE_KEY not in repr(result)
    assert config_calls[0][0] == workspace


def test_upstream_error_cannot_leak_fake_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_calls: list[tuple[Path, dict[str, object]]] = []

    async def leaking_failure(**kwargs: object) -> tuple[str, object]:
        raise RuntimeError(f"unsafe upstream detail {FAKE_KEY}")

    monkeypatch.setattr(
        microsoft,
        "_load_microsoft_bindings",
        lambda: _bindings(
            local_search=leaking_failure,
            global_search=leaking_failure,
            config_calls=config_calls,
        ),
    )
    backend = MicrosoftGraphRAGBackend(
        root_dir=WORKSPACE,
        artifact_dir=FIXTURE_DIR,
        environment={"GRAPHRAG_API_KEY": FAKE_KEY},
    )

    with pytest.raises(GraphRAGQueryError) as captured:
        asyncio.run(backend.query("A live question", mode=GraphQueryMode.LOCAL))

    assert str(captured.value) == "Microsoft GraphRAG LOCAL query failed"
    assert FAKE_KEY not in str(captured.value)
    assert FAKE_KEY not in repr(captured.value)
    assert FAKE_KEY not in caplog.text


def test_default_runtime_has_no_graphrag_route() -> None:
    runtime_source = (PROJECT_ROOT / "app" / "runtime.py").read_text(encoding="utf-8")

    assert "graph_rag" not in runtime_source.lower()
    assert "HybridFinancialRetriever" in runtime_source
