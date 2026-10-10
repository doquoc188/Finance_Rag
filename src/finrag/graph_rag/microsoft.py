"""Lazy Microsoft GraphRAG v3.3 adapter for explicit live queries."""

from __future__ import annotations

import importlib
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from .artifacts import GraphArtifacts, load_graph_artifacts
from .contracts import GraphProvenance, GraphQueryMode, GraphQueryResult
from .errors import (
    GraphRAGConfigurationError,
    GraphRAGDependencyError,
    GraphRAGQueryError,
)

_API_KEY_NAME = "GRAPHRAG_API_KEY"
_RESPONSE_TYPE = "Multiple Paragraphs"

type SearchResult = tuple[object, object]
type SearchFunction = Callable[..., Awaitable[SearchResult]]


@dataclass(frozen=True, slots=True)
class _MicrosoftBindings:
    dataframe: Callable[[list[dict[str, object]]], object]
    load_config: Callable[..., object]
    local_search: SearchFunction
    global_search: SearchFunction


def _load_microsoft_bindings() -> _MicrosoftBindings:
    """Import the optional package only after live preflight succeeds."""

    pandas = importlib.import_module("pandas")
    api = importlib.import_module("graphrag.api")
    config_module = importlib.import_module("graphrag.config.load_config")
    return _MicrosoftBindings(
        dataframe=pandas.DataFrame,
        load_config=config_module.load_config,
        local_search=api.local_search,
        global_search=api.global_search,
    )


def _credential_configured(root_dir: Path, environment: Mapping[str, str] | None) -> bool:
    active_environment = os.environ if environment is None else environment
    value = active_environment.get(_API_KEY_NAME)
    if isinstance(value, str) and value.strip():
        return True
    if environment is not None:
        return False
    workspace_env = root_dir / ".env"
    if not workspace_env.is_file():
        return False
    workspace_value = dotenv_values(workspace_env).get(_API_KEY_NAME)
    return isinstance(workspace_value, str) and bool(workspace_value.strip())


def _resolve_bindings() -> _MicrosoftBindings:
    try:
        return _load_microsoft_bindings()
    except ModuleNotFoundError:
        raise GraphRAGDependencyError(
            "Microsoft GraphRAG is not installed; install the project with the graphrag extra"
        ) from None
    except Exception:
        # Import-time failures from the optional stack can contain environment or request
        # details. Keep that third-party text outside the project exception boundary.
        raise GraphRAGQueryError("Microsoft GraphRAG could not be initialized") from None


def _entity_rows(artifacts: GraphArtifacts) -> list[dict[str, object]]:
    return [
        {
            "id": item.id,
            "human_readable_id": item.id,
            "title": item.title,
            "type": item.type,
            "description": item.description,
            "description_embedding": None,
            "degree": sum(
                relation.source == item.id or relation.target == item.id
                for relation in artifacts.relationships
            ),
            "text_unit_ids": list(item.text_unit_ids),
        }
        for item in artifacts.entities
    ]


def _relationship_rows(artifacts: GraphArtifacts) -> list[dict[str, object]]:
    titles = {item.id: item.title for item in artifacts.entities}
    degree = {
        item.id: sum(
            relation.source == item.id or relation.target == item.id
            for relation in artifacts.relationships
        )
        for item in artifacts.entities
    }
    return [
        {
            "id": item.id,
            "human_readable_id": item.id,
            "source": titles[item.source],
            "target": titles[item.target],
            "description": item.description,
            "weight": item.weight,
            "combined_degree": degree[item.source] + degree[item.target],
            "text_unit_ids": list(item.text_unit_ids),
        }
        for item in artifacts.relationships
    ]


def _text_unit_rows(artifacts: GraphArtifacts) -> list[dict[str, object]]:
    return [
        {
            "id": item.id,
            "text": item.text,
            "document_id": item.document_id,
            "entity_ids": list(item.entity_ids),
            "relationship_ids": list(item.relationship_ids),
            "covariate_ids": None,
            "n_tokens": item.n_tokens,
        }
        for item in artifacts.text_units
    ]


def _community_rows(artifacts: GraphArtifacts) -> list[dict[str, object]]:
    return [
        {
            "id": item.id,
            "community": item.community,
            "title": item.title,
            "level": item.level,
            "parent": item.parent,
            "children": list(item.children),
            "entity_ids": list(item.entity_ids),
            "relationship_ids": list(item.relationship_ids),
            "text_unit_ids": list(item.text_unit_ids),
        }
        for item in artifacts.communities
    ]


def _report_rows(artifacts: GraphArtifacts) -> list[dict[str, object]]:
    return [
        {
            "id": item.id,
            "community": item.community,
            "title": item.title,
            "summary": item.summary,
            "full_content": item.full_content,
            "rank": item.rank,
            "level": item.level,
        }
        for item in artifacts.community_reports
    ]


def _context_records(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        records = to_dict(orient="records")
        if isinstance(records, list):
            return [item for item in records if isinstance(item, Mapping)]
    return []


def _record_identity(record: Mapping[str, object]) -> str | None:
    for key in ("id", "entity", "title", "community"):
        value = record.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value)
    source = record.get("source")
    target = record.get("target")
    if isinstance(source, str) and isinstance(target, str):
        return f"{source}->{target}"
    return None


def _documents_for_record(
    context_type: str, record: Mapping[str, object], artifacts: GraphArtifacts
) -> tuple[str, ...]:
    identifier = _record_identity(record)
    entity_by_id = {item.id: item for item in artifacts.entities}
    entity_by_title = {item.title: item for item in artifacts.entities}
    relationship_by_id = {item.id: item for item in artifacts.relationships}
    report_by_id = {item.id: item for item in artifacts.community_reports}
    communities = {item.community: item for item in artifacts.communities}
    text_units: tuple[str, ...] = ()

    if context_type == "entities" and identifier is not None:
        entity = entity_by_id.get(identifier) or entity_by_title.get(identifier)
        if entity is not None:
            text_units = entity.text_unit_ids
    elif context_type == "relationships" and identifier is not None:
        relationship = relationship_by_id.get(identifier)
        if relationship is not None:
            text_units = relationship.text_unit_ids
    elif context_type == "reports" and identifier is not None:
        report = report_by_id.get(identifier)
        if report is None:
            try:
                report = next(
                    item
                    for item in artifacts.community_reports
                    if item.community == int(identifier)
                )
            except (StopIteration, ValueError):
                report = None
        if report is not None and report.community in communities:
            text_units = communities[report.community].text_unit_ids
    elif context_type == "sources" and identifier is not None:
        text_units = (identifier,)
    return artifacts.document_ids_for_text_units(text_units)


def _translate_provenance(
    context: object, artifacts: GraphArtifacts
) -> tuple[GraphProvenance, ...]:
    if not isinstance(context, Mapping):
        return ()
    provenance = []
    seen: set[tuple[str, str]] = set()
    for raw_type, value in context.items():
        context_type = str(raw_type).lower()
        for record in _context_records(value):
            record_id = _record_identity(record)
            if record_id is None or (context_type, record_id) in seen:
                continue
            document_ids = _documents_for_record(context_type, record, artifacts)
            if document_ids:
                provenance.append(GraphProvenance(context_type, record_id, document_ids))
                seen.add((context_type, record_id))
    return tuple(provenance)


class MicrosoftGraphRAGBackend:
    """Explicit Microsoft GraphRAG LOCAL/GLOBAL adapter with no hybrid fallback."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        artifact_dir: str | Path,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self._root_dir = Path(root_dir)
        self._artifact_dir = Path(artifact_dir)
        self._environment = environment

    async def query(self, question: str, *, mode: GraphQueryMode) -> GraphQueryResult:
        """Validate first, then dispatch only to the caller-selected Microsoft API."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonblank string")
        if not isinstance(mode, GraphQueryMode):
            raise TypeError("mode must be a GraphQueryMode")
        artifacts = load_graph_artifacts(self._artifact_dir, mode=mode)
        settings_path = self._root_dir / "settings.yaml"
        if not settings_path.is_file():
            raise GraphRAGConfigurationError("The GraphRAG workspace must contain settings.yaml")
        if not _credential_configured(self._root_dir, self._environment):
            raise GraphRAGConfigurationError(
                "GRAPHRAG_API_KEY is required for the requested live GraphRAG operation"
            )

        bindings = _resolve_bindings()
        frames = {
            "entities": bindings.dataframe(_entity_rows(artifacts)),
            "relationships": bindings.dataframe(_relationship_rows(artifacts)),
            "text_units": bindings.dataframe(_text_unit_rows(artifacts)),
            "communities": bindings.dataframe(_community_rows(artifacts)),
            "community_reports": bindings.dataframe(_report_rows(artifacts)),
        }
        overrides = {
            "vector_store": {"db_uri": str(self._root_dir / "output" / "lancedb")},
            "reporting": {"base_dir": str(self._root_dir / "logs")},
        }
        try:
            configuration = bindings.load_config(self._root_dir, cli_overrides=overrides)
            if mode is GraphQueryMode.LOCAL:
                answer, context = await bindings.local_search(
                    config=configuration,
                    entities=frames["entities"],
                    communities=frames["communities"],
                    community_reports=frames["community_reports"],
                    text_units=frames["text_units"],
                    relationships=frames["relationships"],
                    covariates=None,
                    community_level=0,
                    response_type=_RESPONSE_TYPE,
                    query=question.strip(),
                    verbose=False,
                )
            else:
                answer, context = await bindings.global_search(
                    config=configuration,
                    entities=frames["entities"],
                    communities=frames["communities"],
                    community_reports=frames["community_reports"],
                    community_level=0,
                    dynamic_community_selection=False,
                    response_type=_RESPONSE_TYPE,
                    query=question.strip(),
                    verbose=False,
                )
        except Exception:
            # Upstream exception text can include request/configuration details. Preserve a
            # stable project error without retaining the exception as a displayed cause.
            raise GraphRAGQueryError(
                f"Microsoft GraphRAG {mode.value.upper()} query failed"
            ) from None
        if not isinstance(answer, str) or not answer.strip():
            raise GraphRAGQueryError(
                f"Microsoft GraphRAG {mode.value.upper()} returned no text answer"
            )
        return GraphQueryResult(
            answer=answer,
            mode=mode,
            provenance=_translate_provenance(context, artifacts),
        )
