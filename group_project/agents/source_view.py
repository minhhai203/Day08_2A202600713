from __future__ import annotations

from textwrap import shorten

from .contracts import Citation, SourceDocument


def format_citation_line(citation: Citation) -> str:
    parts = [f"**[{citation.id}] {citation.title}**"]
    if citation.source:
        parts.append(f"Source: {citation.source}")
    if citation.url:
        parts.append(f"Link: {citation.url}")
    return " | ".join(parts)


def format_citations(citations: list[Citation]) -> str:
    if not citations:
        return "_Chưa có citation._"
    lines = ["### Citation"]
    lines.extend(f"- {format_citation_line(citation)}" for citation in citations)
    return "\n".join(lines)


def format_source_card(source: SourceDocument) -> str:
    parts = [f"**{source.title}**"]
    if source.source:
        parts.append(f"Source: {source.source}")
    if source.snippet:
        snippet = shorten(source.snippet.replace("\n", " "), width=180, placeholder="...")
        parts.append(f"> {snippet}")
    if source.url:
        parts.append(f"Link: {source.url}")
    if source.score is not None:
        parts.append(f"Score: {source.score:.3f}")
    return "\n".join(parts)


def format_sources_panel(sources: list[SourceDocument]) -> str:
    if not sources:
        return "_Chưa có source documents._"
    lines = ["### Source Documents"]
    for idx, source in enumerate(sources, start=1):
        lines.append(f"#### Source {idx}")
        lines.append(format_source_card(source))
    return "\n\n".join(lines)


def format_sources_sidebar(
    sources: list[SourceDocument],
    citations: list[Citation] | None = None,
    retrieval_mode: str | None = None,
    used_memory: bool | None = None,
) -> str:
    lines: list[str] = ["# Source Panel"]
    if retrieval_mode:
        lines.append(f"- Retrieval: {retrieval_mode}")
    if used_memory is not None:
        lines.append(f"- Memory: {'on' if used_memory else 'off'}")
    if citations:
        lines.append(f"- Citations: {len(citations)}")

    if not sources:
        lines.append("\n_No source documents available yet._")
        return "\n".join(lines)

    lines.append("\n## Top Sources")
    for idx, source in enumerate(sources, start=1):
        lines.append(f"### {idx}. {source.title}")
        if source.source:
            lines.append(f"- Source: {source.source}")
        if source.score is not None:
            lines.append(f"- Score: {source.score:.3f}")
        if source.url:
            lines.append(f"- Link: {source.url}")
        if source.snippet:
            snippet = shorten(source.snippet.replace("\n", " "), width=220, placeholder="...")
            lines.append(f"- Snippet: {snippet}")
    return "\n".join(lines)
