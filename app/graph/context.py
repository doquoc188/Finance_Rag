"""Small deterministic evidence formatting for the existing Document runtime."""

from collections.abc import Sequence

from langchain_core.documents import Document


def serialize_evidence(documents: Sequence[Document]) -> str:
    """Retain document order/text and available scalar source metadata only."""
    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        lines = [f"[SOURCE {index}]"]
        for key in ("source", "page", "company", "year", "has_table"):
            value = document.metadata.get(key)
            if isinstance(value, (str, int, float, bool)):
                lines.append(f"{key}: {value}")
        lines.append(f"content:\n{document.page_content}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)
