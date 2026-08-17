"""Convert retrieval matches into expanded context snippets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from src.config.retrieval_config import INCLUDE_SIBLINGS, SNIPPET_RETURN_MODE, STITCH_MAX_GAP
from src.parsing.document_types import (
    ContextSnippet,
    NodeKind,
    RetrievalMatch,
    StoredNode,
    public_metadata,
)


class NodeStore(Protocol):
    async def get_context_batch(
        self,
        specs: Sequence[tuple[str, int, int]],
        *,
        include_siblings: bool = False,
    ) -> list[list[StoredNode]]: ...

    async def get_nodes(self, node_ids: Sequence[str]) -> dict[str, StoredNode]: ...


def _expand_parent_depth(
    match: RetrievalMatch,
    parent_depth: int,
) -> int:
    """Only sentences may walk up, and only to their paragraph.

    Lifting a paragraph to its section is harmful on heading-poor PDFs:
    the whole document is one section, tree-dedup then drops every child,
    and the API returns a truncated section summary instead of the hits.
    """
    if parent_depth <= 0:
        return 0
    if match.node.kind == NodeKind.SENTENCE:
        return 1
    return 0


def _select_primary_node(
    match: RetrievalMatch,
    nodes: Sequence[StoredNode],
) -> StoredNode:
    """parent_child：句子命中提升为所属段落（小块召回、大块返回）。"""
    by_id = {node.node_id: node for node in nodes}
    if SNIPPET_RETURN_MODE == "parent_child" and match.node.kind == NodeKind.SENTENCE:
        parent = by_id.get(match.node.parent_id or "")
        if parent is not None and parent.kind == NodeKind.PARAGRAPH:
            return parent
    return by_id.get(match.node.node_id, match.node)


def _skip_as_context(node: StoredNode) -> bool:
    return SNIPPET_RETURN_MODE == "parent_child" and node.kind in {
        NodeKind.SECTION,
        NodeKind.DOCUMENT,
    }


def _ordered_context_nodes(
    match: RetrievalMatch,
    nodes: Sequence[StoredNode],
) -> list[StoredNode]:
    primary = _select_primary_node(match, nodes)
    ordered = [primary] + [node for node in nodes if node.node_id != primary.node_id]
    return [node for node in ordered if not _skip_as_context(node)]


async def expand_matches_to_retrieval(
    matches: Sequence[RetrievalMatch],
    store: NodeStore,
    *,
    parent_depth: int = 1,
    child_depth: int = 0,
    include_siblings: bool | None = None,
    limit: int | None = None,
) -> list[RetrievalMatch]:
    """把命中扩展为独立节点（含邻近兄弟），供 rerank 逐条打分。"""
    if not matches:
        return []

    siblings = INCLUDE_SIBLINGS if include_siblings is None else include_siblings
    specs = [
        (
            match.node.node_id,
            _expand_parent_depth(match, parent_depth),
            child_depth,
        )
        for match in matches
    ]
    context_batches = await store.get_context_batch(
        specs,
        include_siblings=siblings,
    )

    expanded: list[RetrievalMatch] = []
    seen: set[str] = set()
    for match, nodes in zip(matches, context_batches, strict=True):
        for node in _ordered_context_nodes(match, nodes):
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            expanded.append(RetrievalMatch(node=node, score=match.score))
            if limit is not None and len(expanded) >= limit:
                return expanded
    return expanded


def matches_as_snippets(matches: Sequence[RetrievalMatch]) -> list[ContextSnippet]:
    """Rerank 之后不再二次扩邻居：命中列表即上下文。"""
    snippets: list[ContextSnippet] = []
    seen: set[str] = set()
    for rank, match in enumerate(matches, 1):
        node = match.node
        if node.node_id in seen or _skip_as_context(node):
            continue
        seen.add(node.node_id)
        snippets.append(
            ContextSnippet(
                node_id=node.node_id,
                document_title=node.metadata.get("document_title", node.title),
                text=node.text,
                metadata=public_metadata(node.metadata),
                rank=rank,
                score=match.score,
            )
        )
    return snippets


_PARAGRAPH_ID = re.compile(r":p(\d+)(?:$|:)")
_TERMINATORS = "。！？；：.!?…"


@dataclass
class _IndexedHit:
    match: RetrievalMatch
    index: int
    filled: bool = False


def _paragraph_index(node: StoredNode) -> int | None:
    raw = (node.metadata or {}).get("paragraph_index")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    found = _PARAGRAPH_ID.search(node.node_id)
    return int(found.group(1)) if found else None


def _group_key(node: StoredNode) -> tuple[str, str]:
    meta = node.metadata or {}
    document_id = str(meta.get("document_id") or "")
    section_id = str(meta.get("section_id") or node.parent_id or "")
    return document_id, section_id


def _paragraph_node_id(node_id: str, paragraph_index: int) -> str:
    return _PARAGRAPH_ID.sub(f":p{paragraph_index}", node_id, count=1)


def _join_cluster_texts(texts: Sequence[str]) -> str:
    parts: list[str] = []
    for raw in texts:
        text = (raw or "").strip()
        if not text:
            continue
        if not parts:
            parts.append(text)
            continue
        prev = parts[-1]
        if prev[-1] not in _TERMINATORS and not text.startswith(("➢", "•", "·")):
            parts[-1] = prev + text
        else:
            parts.append(text)
    return "\n".join(parts)


def _cluster_score(cluster: Sequence[_IndexedHit]) -> float:
    return max((item.match.score for item in cluster if not item.filled), default=0.0)


def _hits_to_snippet(cluster: Sequence[_IndexedHit], rank: int) -> ContextSnippet:
    ordered = sorted(cluster, key=lambda item: item.index if item.index >= 0 else 0)
    first = ordered[0].match.node
    metadata = public_metadata(first.metadata)
    if len(ordered) > 1:
        metadata["stitched_node_ids"] = [item.match.node.node_id for item in ordered]
    return ContextSnippet(
        node_id=first.node_id,
        document_title=first.metadata.get("document_title", first.title),
        text=_join_cluster_texts([item.match.node.text for item in ordered]),
        metadata=metadata,
        rank=rank,
        score=_cluster_score(ordered),
    )


def _split_into_runs(ordered: list[_IndexedHit], max_gap: int) -> list[list[_IndexedHit]]:
    if not ordered:
        return []
    runs: list[list[_IndexedHit]] = [[ordered[0]]]
    for hit in ordered[1:]:
        if hit.index - runs[-1][-1].index <= 1 + max_gap:
            runs[-1].append(hit)
        else:
            runs.append([hit])
    return runs


async def _fill_run(
    run: list[_IndexedHit],
    store: NodeStore,
) -> list[_IndexedHit]:
    if len(run) < 2:
        return run
    missing: list[int] = []
    for left, right in zip(run, run[1:]):
        missing.extend(range(left.index + 1, right.index))
    if not missing:
        return run
    node_ids = [_paragraph_node_id(run[0].match.node.node_id, index) for index in missing]
    fetched = await store.get_nodes(node_ids)
    by_index = {item.index: item for item in run}
    filled: list[_IndexedHit] = []
    start, end = run[0].index, run[-1].index
    inherit_score = run[0].match.score
    for index in range(start, end + 1):
        existing = by_index.get(index)
        if existing is not None:
            filled.append(existing)
            continue
        node = fetched.get(_paragraph_node_id(run[0].match.node.node_id, index))
        if node is None or _skip_as_context(node):
            return run
        filled.append(
            _IndexedHit(
                RetrievalMatch(node=node, score=inherit_score),
                index,
                filled=True,
            )
        )
    return filled


async def stitch_consecutive_matches(
    matches: Sequence[RetrievalMatch],
    store: NodeStore,
    *,
    max_gap: int | None = None,
) -> list[ContextSnippet]:
    """把同文档、连续（或只缺一段）的命中按阅读顺序拼成给模型的 snippet。"""
    gap = STITCH_MAX_GAP if max_gap is None else max_gap
    grouped: dict[tuple[str, str], list[_IndexedHit]] = {}
    singles: list[_IndexedHit] = []
    for match in matches:
        if _skip_as_context(match.node):
            continue
        if match.node.kind != NodeKind.PARAGRAPH:
            singles.append(_IndexedHit(match, -1))
            continue
        index = _paragraph_index(match.node)
        if index is None:
            singles.append(_IndexedHit(match, -1))
            continue
        grouped.setdefault(_group_key(match.node), []).append(_IndexedHit(match, index))

    clusters: list[list[_IndexedHit]] = [[item] for item in singles]
    for hits in grouped.values():
        unique: dict[int, _IndexedHit] = {}
        for hit in hits:
            previous = unique.get(hit.index)
            if previous is None or hit.match.score > previous.match.score:
                unique[hit.index] = hit
        ordered = [unique[index] for index in sorted(unique)]
        for run in _split_into_runs(ordered, gap):
            clusters.append(await _fill_run(run, store))

    clusters.sort(key=_cluster_score, reverse=True)
    return [_hits_to_snippet(cluster, rank) for rank, cluster in enumerate(clusters, 1)]


async def matches_to_snippets(
    matches: Sequence[RetrievalMatch],
    store: NodeStore,
    *,
    parent_depth: int = 1,
    child_depth: int = 0,
    dedup: str = "none",
    include_siblings: bool | None = None,
) -> list[ContextSnippet]:
    """Expand each match into parent/child/sibling context and optionally deduplicate."""
    expanded = await expand_matches_to_retrieval(
        matches,
        store,
        parent_depth=parent_depth,
        child_depth=child_depth,
        include_siblings=include_siblings,
    )
    snippets = [
        ContextSnippet(
            node_id=item.node.node_id,
            document_title=item.node.metadata.get("document_title", item.node.title),
            text=item.node.text,
            metadata=public_metadata(item.node.metadata),
            rank=rank,
            score=item.score,
        )
        for rank, item in enumerate(expanded, 1)
    ]

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
