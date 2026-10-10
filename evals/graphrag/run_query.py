"""Validate the curated graph offline or explicitly run one live GraphRAG query."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from finrag.graph_rag import (
    GraphQueryMode,
    GraphRAGError,
    MicrosoftGraphRAGBackend,
    load_graph_artifacts,
)

EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = EVAL_ROOT / "fixtures" / "curated_finance"
DEFAULT_WORKSPACE = EVAL_ROOT / "workspace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate curated artifacts offline; add --live for a paid Microsoft query."
    )
    parser.add_argument("--mode", required=True, choices=[mode.value for mode in GraphQueryMode])
    parser.add_argument("--question", required=True, help="Question used only when --live is set")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing the curated JSON artifact tables",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Microsoft GraphRAG workspace containing settings.yaml",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Explicitly permit a hosted Microsoft GraphRAG query that may incur API cost",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    mode = GraphQueryMode(args.mode)
    artifacts = load_graph_artifacts(args.artifact_dir, mode=mode)
    print(
        "Validated synthetic graph: "
        f"documents={len(artifacts.documents)}, entities={len(artifacts.entities)}, "
        f"relationships={len(artifacts.relationships)}, text_units={len(artifacts.text_units)}, "
        f"community_reports={len(artifacts.community_reports)}"
    )
    if not args.live:
        print("Offline validation complete. REAL GRAPHRAG QUERY: NOT EXECUTED")
        return 0

    backend = MicrosoftGraphRAGBackend(
        root_dir=args.workspace,
        artifact_dir=args.artifact_dir,
    )
    result = await backend.query(args.question, mode=mode)
    print(f"Mode: {result.mode.value}")
    print(result.answer)
    if result.provenance:
        print("Source document IDs:")
        document_ids = dict.fromkeys(
            document_id for item in result.provenance for document_id in item.document_ids
        )
        for document_id in document_ids:
            print(f"- {document_id}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(run(args))
    except (GraphRAGError, ValueError, TypeError) as error:
        raise SystemExit(f"GraphRAG demo failed: {error}") from None


if __name__ == "__main__":
    raise SystemExit(main())
