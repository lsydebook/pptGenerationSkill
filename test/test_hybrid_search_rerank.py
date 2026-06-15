"""Tests for hybrid search rerank score output."""

import unittest

from src.parsing.document_types import NodeKind, RetrievalMatch, StoredNode
from src.retrieval.hybrid_search import deduplicate_matches, rerank_matches


def _node(node_id: str) -> StoredNode:
    return StoredNode(
        node_id=node_id,
        parent_id=None,
        kind=NodeKind.PARAGRAPH,
        title=node_id,
        text=node_id,
        metadata={},
        embedding=[0.0],
    )


class HybridSearchRerankTest(unittest.TestCase):
    def test_rerank_combined_writes_final_score_not_first_hit(self) -> None:
        p3 = _node("doc:sec1:p3")
        p61 = _node("loratadine:sec20:p61")
        matches = [
            RetrievalMatch(node=p3, score=0.117),
            RetrievalMatch(node=p61, score=0.927),
            RetrievalMatch(node=p3, score=1.0),
            RetrievalMatch(node=p3, score=0.244),
        ]

        ranked = rerank_matches(matches, strategy="combined")

        self.assertEqual(ranked[0].node.node_id, "doc:sec1:p3")
        self.assertGreater(ranked[0].score, ranked[1].score)
        self.assertNotEqual(ranked[0].score, 0.117)
        self.assertEqual(ranked[1].node.node_id, "loratadine:sec20:p61")
        self.assertLess(ranked[1].score, ranked[0].score)

    def test_deduplicate_uses_max_score(self) -> None:
        node = _node("doc:sec1:p1")
        matches = [
            RetrievalMatch(node=node, score=0.1),
            RetrievalMatch(node=node, score=0.9),
        ]

        deduped = deduplicate_matches(matches)

        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0].score, 0.9)


if __name__ == "__main__":
    unittest.main()
