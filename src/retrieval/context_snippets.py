"""Convert retrieval matches into expanded context snippets."""

from __future__ import annotations

from typing import Protocol, Sequence

from src.parsing.document_types import ContextSnippet, NodeKind, RetrievalMatch, StoredNode


class NodeStore(Protocol):
    async def get_context(
        self,
        node_id: str,
        *,
        parent_depth: int = 1,
        child_depth: int = 0,
    ) -> list[StoredNode]: ...


async def matches_to_snippets(
    matches: Sequence[RetrievalMatch],
    store: NodeStore,
    *,
    parent_depth: int = 1,
    child_depth: int = 0,
    dedup: str = "none",
) -> list[ContextSnippet]:
    """Expand each match into parent/child context and optionally deduplicate."""
    snippets: list[ContextSnippet] = []
    for rank, match in enumerate(matches, 1):
        # 段落已是完整语义单元，不再上溯到 section（通常只有标题，会挤掉正文）
        expand_parent = parent_depth
        if match.node.kind == NodeKind.PARAGRAPH:
            expand_parent = 0
        elif match.node.kind == NodeKind.SENTENCE and parent_depth > 0:
            expand_parent = min(parent_depth, 1)

        nodes = await store.get_context(
            match.node.node_id,
            parent_depth=expand_parent,
            child_depth=child_depth,
        )
        for context_node in nodes:
            snippets.append(
                ContextSnippet(
                    node_id=context_node.node_id,
                    document_title=context_node.metadata.get(
                        "document_title", context_node.title
                    ),
                    text=context_node.text,
                    metadata=context_node.metadata,
                    rank=rank,
                    score=match.score,
                )
            )

    if dedup == "node_id":
        return _dedup_snippets_by_node_id(snippets)
    if dedup == "tree":
        return _remove_overlapping_snippets(snippets)
    return snippets


def _dedup_snippets_by_node_id(
    snippets: list[ContextSnippet],
) -> list[ContextSnippet]:
    seen: set[str] = set()
    result: list[ContextSnippet] = []
    for snippet in snippets:
        if snippet.node_id not in seen:
            seen.add(snippet.node_id)
            result.append(snippet)
    return result


def _remove_overlapping_snippets(
    snippets: list[ContextSnippet],
) -> list[ContextSnippet]:
    if not snippets:
        return snippets

    all_node_ids = {s.node_id for s in snippets}
    nodes_with_ancestor: set[str] = set()
    for node_id in all_node_ids:
        for other_id in all_node_ids:
            if other_id != node_id and node_id.startswith(other_id + ":"):
                nodes_with_ancestor.add(node_id)
                break

    seen_ids: set[str] = set()
    filtered: list[ContextSnippet] = []
    for snippet in snippets:
        if snippet.node_id in nodes_with_ancestor:
            continue
        if snippet.node_id in seen_ids:
            continue
        seen_ids.add(snippet.node_id)
        filtered.append(snippet)
    return filtered
