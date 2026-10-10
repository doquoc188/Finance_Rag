"""Strict standard-library loading for the curated GraphRAG fixture."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from .contracts import GraphQueryMode
from .errors import GraphRAGArtifactError

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class GraphDocument:
    id: str
    title: str
    source: str
    text: str


@dataclass(frozen=True, slots=True)
class GraphEntity:
    id: str
    title: str
    type: str
    description: str
    text_unit_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphRelationship:
    id: str
    source: str
    target: str
    description: str
    weight: float
    text_unit_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphTextUnit:
    id: str
    text: str
    document_id: str
    entity_ids: tuple[str, ...]
    relationship_ids: tuple[str, ...]
    n_tokens: int


@dataclass(frozen=True, slots=True)
class GraphCommunity:
    id: str
    community: int
    title: str
    level: int
    parent: int
    children: tuple[int, ...]
    entity_ids: tuple[str, ...]
    relationship_ids: tuple[str, ...]
    text_unit_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GraphCommunityReport:
    id: str
    community: int
    title: str
    summary: str
    full_content: str
    rank: float
    level: int


@dataclass(frozen=True, slots=True)
class GraphArtifacts:
    """Validated project-owned graph values ready for adapter translation."""

    documents: tuple[GraphDocument, ...]
    entities: tuple[GraphEntity, ...]
    relationships: tuple[GraphRelationship, ...]
    text_units: tuple[GraphTextUnit, ...]
    communities: tuple[GraphCommunity, ...]
    community_reports: tuple[GraphCommunityReport, ...]

    def document_ids_for_text_units(self, text_unit_ids: Sequence[str]) -> tuple[str, ...]:
        by_id = {item.id: item.document_id for item in self.text_units}
        return tuple(dict.fromkeys(by_id[item_id] for item_id in text_unit_ids if item_id in by_id))


def _fail(message: str) -> GraphRAGArtifactError:
    return GraphRAGArtifactError(message)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise _fail(f"Missing GraphRAG artifact: {path.name}") from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise _fail(f"Invalid GraphRAG JSON artifact: {path.name}") from None


def _record(value: object, *, artifact: str, index: int, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{artifact}[{index}] must be an object")
    unexpected = set(value) - fields
    missing = fields - set(value)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected {sorted(unexpected)}")
        raise _fail(f"{artifact}[{index}] has " + " and ".join(details))
    return value


def _records(root: Path, name: str) -> list[object]:
    value = _load_json(root / f"{name}.json")
    if not isinstance(value, list):
        raise _fail(f"{name}.json must contain an array")
    return value


def _string(record: Mapping[str, Any], field: str, location: str) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise _fail(f"{location}.{field} must be a nonblank string")
    return value


def _strings(record: Mapping[str, Any], field: str, location: str) -> tuple[str, ...]:
    value = record[field]
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise _fail(f"{location}.{field} must be an array of nonblank strings")
    if len(value) != len(set(value)):
        raise _fail(f"{location}.{field} contains duplicate IDs")
    return tuple(value)


def _integer(record: Mapping[str, Any], field: str, location: str, *, minimum: int = 0) -> int:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(f"{location}.{field} must be an integer >= {minimum}")
    return value


def _integers(record: Mapping[str, Any], field: str, location: str) -> tuple[int, ...]:
    value = record[field]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise _fail(f"{location}.{field} must be an array of integers")
    return tuple(value)


def _number(record: Mapping[str, Any], field: str, location: str) -> float:
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise _fail(f"{location}.{field} must be a finite number")
    return float(value)


def _unique_ids(values: Sequence[object], artifact: str) -> None:
    identifiers = [getattr(value, "id") for value in values]
    if len(identifiers) != len(set(identifiers)):
        duplicate = next(item for item in identifiers if identifiers.count(item) > 1)
        raise _fail(f"Duplicate {artifact} ID: {duplicate}")


def _load_documents(root: Path) -> tuple[GraphDocument, ...]:
    fields = {"id", "title", "source", "text"}
    documents = []
    for index, value in enumerate(_records(root, "documents")):
        location = f"documents[{index}]"
        row = _record(value, artifact="documents", index=index, fields=fields)
        documents.append(
            GraphDocument(
                id=_string(row, "id", location),
                title=_string(row, "title", location),
                source=_string(row, "source", location),
                text=_string(row, "text", location),
            )
        )
    _unique_ids(documents, "document")
    return tuple(documents)


def _load_entities(root: Path) -> tuple[GraphEntity, ...]:
    fields = {"id", "title", "type", "description", "text_unit_ids"}
    entities = []
    for index, value in enumerate(_records(root, "entities")):
        location = f"entities[{index}]"
        row = _record(value, artifact="entities", index=index, fields=fields)
        entities.append(
            GraphEntity(
                id=_string(row, "id", location),
                title=_string(row, "title", location),
                type=_string(row, "type", location),
                description=_string(row, "description", location),
                text_unit_ids=_strings(row, "text_unit_ids", location),
            )
        )
    if not entities:
        raise _fail("GraphRAG entities must not be empty")
    _unique_ids(entities, "entity")
    titles = [entity.title for entity in entities]
    if len(titles) != len(set(titles)):
        raise _fail("GraphRAG entity titles must be unique")
    return tuple(entities)


def _load_relationships(root: Path) -> tuple[GraphRelationship, ...]:
    fields = {"id", "source", "target", "description", "weight", "text_unit_ids"}
    relationships = []
    for index, value in enumerate(_records(root, "relationships")):
        location = f"relationships[{index}]"
        row = _record(value, artifact="relationships", index=index, fields=fields)
        relationships.append(
            GraphRelationship(
                id=_string(row, "id", location),
                source=_string(row, "source", location),
                target=_string(row, "target", location),
                description=_string(row, "description", location),
                weight=_number(row, "weight", location),
                text_unit_ids=_strings(row, "text_unit_ids", location),
            )
        )
    if not relationships:
        raise _fail("GraphRAG relationships must not be empty")
    _unique_ids(relationships, "relationship")
    return tuple(relationships)


def _load_text_units(root: Path) -> tuple[GraphTextUnit, ...]:
    fields = {"id", "text", "document_id", "entity_ids", "relationship_ids", "n_tokens"}
    text_units = []
    for index, value in enumerate(_records(root, "text_units")):
        location = f"text_units[{index}]"
        row = _record(value, artifact="text_units", index=index, fields=fields)
        text_units.append(
            GraphTextUnit(
                id=_string(row, "id", location),
                text=_string(row, "text", location),
                document_id=_string(row, "document_id", location),
                entity_ids=_strings(row, "entity_ids", location),
                relationship_ids=_strings(row, "relationship_ids", location),
                n_tokens=_integer(row, "n_tokens", location, minimum=1),
            )
        )
    _unique_ids(text_units, "text unit")
    return tuple(text_units)


def _load_communities(root: Path) -> tuple[GraphCommunity, ...]:
    fields = {
        "id",
        "community",
        "title",
        "level",
        "parent",
        "children",
        "entity_ids",
        "relationship_ids",
        "text_unit_ids",
    }
    communities = []
    for index, value in enumerate(_records(root, "communities")):
        location = f"communities[{index}]"
        row = _record(value, artifact="communities", index=index, fields=fields)
        communities.append(
            GraphCommunity(
                id=_string(row, "id", location),
                community=_integer(row, "community", location),
                title=_string(row, "title", location),
                level=_integer(row, "level", location),
                parent=_integer(row, "parent", location, minimum=-1),
                children=_integers(row, "children", location),
                entity_ids=_strings(row, "entity_ids", location),
                relationship_ids=_strings(row, "relationship_ids", location),
                text_unit_ids=_strings(row, "text_unit_ids", location),
            )
        )
    _unique_ids(communities, "community")
    if len({item.community for item in communities}) != len(communities):
        raise _fail("Duplicate GraphRAG community number")
    return tuple(communities)


def _load_reports(root: Path) -> tuple[GraphCommunityReport, ...]:
    fields = {"id", "community", "title", "summary", "full_content", "rank", "level"}
    reports = []
    for index, value in enumerate(_records(root, "community_reports")):
        location = f"community_reports[{index}]"
        row = _record(value, artifact="community_reports", index=index, fields=fields)
        reports.append(
            GraphCommunityReport(
                id=_string(row, "id", location),
                community=_integer(row, "community", location),
                title=_string(row, "title", location),
                summary=_string(row, "summary", location),
                full_content=_string(row, "full_content", location),
                rank=_number(row, "rank", location),
                level=_integer(row, "level", location),
            )
        )
    _unique_ids(reports, "community report")
    return tuple(reports)


def _validate_references(artifacts: GraphArtifacts, mode: GraphQueryMode | None) -> None:
    document_ids = {item.id for item in artifacts.documents}
    entity_ids = {item.id for item in artifacts.entities}
    relationship_ids = {item.id for item in artifacts.relationships}
    text_unit_ids = {item.id for item in artifacts.text_units}
    community_numbers = {item.community for item in artifacts.communities}

    if (mode is None or mode is GraphQueryMode.LOCAL) and not artifacts.text_units:
        raise _fail("GraphRAG LOCAL artifacts require text units")
    if (mode is None or mode is GraphQueryMode.GLOBAL) and not artifacts.community_reports:
        raise _fail("GraphRAG GLOBAL artifacts require community reports")
    for relationship in artifacts.relationships:
        unknown = {relationship.source, relationship.target} - entity_ids
        if unknown:
            raise _fail(
                f"Relationship {relationship.id} points to unknown entity IDs: {sorted(unknown)}"
            )
        if not set(relationship.text_unit_ids) <= text_unit_ids:
            raise _fail(f"Relationship {relationship.id} references unknown text unit IDs")
    for entity in artifacts.entities:
        if not set(entity.text_unit_ids) <= text_unit_ids:
            raise _fail(f"Entity {entity.id} references unknown text unit IDs")
    for text_unit in artifacts.text_units:
        if text_unit.document_id not in document_ids:
            raise _fail(f"Text unit {text_unit.id} references an unknown document ID")
        if not set(text_unit.entity_ids) <= entity_ids:
            raise _fail(f"Text unit {text_unit.id} references unknown entity IDs")
        if not set(text_unit.relationship_ids) <= relationship_ids:
            raise _fail(f"Text unit {text_unit.id} references unknown relationship IDs")
    for community in artifacts.communities:
        if not set(community.entity_ids) <= entity_ids:
            raise _fail(f"Community {community.id} references unknown entity IDs")
        if not set(community.relationship_ids) <= relationship_ids:
            raise _fail(f"Community {community.id} references unknown relationship IDs")
        if not set(community.text_unit_ids) <= text_unit_ids:
            raise _fail(f"Community {community.id} references unknown text unit IDs")
    for report in artifacts.community_reports:
        if report.community not in community_numbers:
            raise _fail(f"Community report {report.id} references an unknown community")


def load_graph_artifacts(root: str | Path, *, mode: GraphQueryMode | None = None) -> GraphArtifacts:
    """Load and validate the deterministic curated artifact directory."""

    root_path = Path(root)
    manifest = _load_json(root_path / "manifest.json")
    if manifest != {
        "description": "Synthetic curated finance graph for GraphRAG integration tests",
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
    }:
        raise _fail("GraphRAG fixture manifest is missing or unsupported")
    artifacts = GraphArtifacts(
        documents=_load_documents(root_path),
        entities=_load_entities(root_path),
        relationships=_load_relationships(root_path),
        text_units=_load_text_units(root_path),
        communities=_load_communities(root_path),
        community_reports=_load_reports(root_path),
    )
    _validate_references(artifacts, mode)
    return artifacts
