from __future__ import annotations

from .contracts import Citation, SourceDocument


def format_citation_line(citation: Citation) -> str:
    parts = [f"[{citation.id}] {citation.title}"]
    if citation.source:
        parts.append(f"Source: {citation.source}")
    if citation.url:
        parts.append(f"Link: {citation.url}")
    return " | ".join(parts)


def format_citations(citations: list[Citation]) -> str:
    if not citations:
        return "_Chưa có citation._"
    return "\n".join(f"- {format_citation_line(citation)}" for citation in citations)


def format_source_card(source: SourceDocument) -> str:
    parts = [f"**{source.title}**"]
    if source.source:
        parts.append(f"Source: {source.source}")
    if source.snippet:
        parts.append(f"> {source.snippet}")
    if source.url:
        parts.append(f"Link: {source.url}")
    if source.score is not None:
        parts.append(f"Score: {source.score:.3f}")
    return "\n".join(parts)


def format_sources_panel(sources: list[SourceDocument]) -> str:
    if not sources:
        return "_Chưa có source documents._"
    return "\n\n".join(f"- {format_source_card(source)}" for source in sources)

