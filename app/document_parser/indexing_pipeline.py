"""Complete indexing pipeline: Parse → Embed → Store."""

from __future__ import annotations

import uuid
from typing import Literal, Sequence

import numpy as np

from .datastore_milvus_pg import MilvusPostgresNodeStore
from .indexer import DocumentIndexer as TreeIndexer
from .jina_embedder import JinaV4Embedder
from .parsers import text_to_payload
from .types import DocumentPayload, NodeKind, StoredNode, TreeNode

ParagraphEmbeddingMode = Literal["averaged", "full", "both"]


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def average_embeddings(child_vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not child_vectors:
        raise ValueError("average_embeddings requires at least one child vector.")
    stacked = np.vstack(child_vectors)
    return _normalize(np.mean(stacked, axis=0, keepdims=True))[0]


class RAGIndexer:
    """Parse hierarchical documents, embed with Jina V4, persist to Milvus + PostgreSQL."""

    def __init__(
        self,
        embedding_model: JinaV4Embedder,
        datastore: MilvusPostgresNodeStore,
        paragraph_embedding_mode: ParagraphEmbeddingMode = "averaged",
    ) -> None:
        self._embedding_model = embedding_model
        self._datastore = datastore
        self._paragraph_embedding_mode = paragraph_embedding_mode
        self._tree_indexer = TreeIndexer()

    async def index_text(
        self,
        text: str,
        *,
        document_id: str | None = None,
        title: str = "Untitled",
        metadata: dict | None = None,
    ) -> list[StoredNode]:
        payload = text_to_payload(
            document_id=document_id or str(uuid.uuid4()),
            title=title,
            text=text,
            metadata=dict(metadata or {}),
        )
        return await self.index_document(payload)

    async def index_document(self, document: DocumentPayload) -> list[StoredNode]:
        print(f"\n[Indexing] Starting: {document.title}")

        root = self._tree_indexer.build_tree(document)

        print("[Indexing] Computing embeddings...")
        await self._embed_tree(root)

        stored_nodes = [self._to_stored(node) for node in self._tree_indexer.flatten(root)]

        print(f"[Indexing] Persisting {len(stored_nodes)} nodes...")
        await self._datastore.upsert_nodes(stored_nodes)

        print(f"[Indexing] Complete: {len(stored_nodes)} nodes\n")
        return stored_nodes

    async def _embed_tree(self, root: TreeNode) -> None:
        leaves = [node for node in self._tree_indexer.flatten(root) if not node.children]
        if leaves:
            print(f"  [Embedding] Encoding {len(leaves)} leaf nodes...")
            embeddings = await self._embedding_model.embed_text(
                [leaf.text for leaf in leaves]
            )
            for leaf, vector in zip(leaves, embeddings, strict=True):
                leaf.embedding = vector

        paragraphs = [
            node
            for node in self._tree_indexer.flatten(root)
            if node.kind == NodeKind.PARAGRAPH
        ]

        para_full_map: dict[str, np.ndarray] = {}
        if self._paragraph_embedding_mode in ("full", "both") and paragraphs:
            print(f"  [Embedding] Encoding {len(paragraphs)} paragraph embeddings...")
            para_embeddings = await self._embedding_model.embed_text(
                [p.text for p in paragraphs]
            )
            para_full_map = {
                p.node_id: vec
                for p, vec in zip(paragraphs, para_embeddings, strict=True)
            }

        self._propagate_embeddings(root, para_full_map)

    def _propagate_embeddings(
        self,
        node: TreeNode,
        para_full_map: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        if para_full_map is None:
            para_full_map = {}

        if node.embedding is not None:
            return node.embedding

        child_vectors = [
            self._propagate_embeddings(child, para_full_map) for child in node.children
        ]
        if not child_vectors:
            raise ValueError(f"Node {node.node_id} is missing an embedding.")

        averaged_embedding = average_embeddings(child_vectors)

        if node.kind == NodeKind.PARAGRAPH and node.node_id in para_full_map:
            full_embedding = para_full_map[node.node_id]
            if self._paragraph_embedding_mode == "full":
                node.embedding = full_embedding
            elif self._paragraph_embedding_mode == "both":
                node.embedding = averaged_embedding
                node.metadata["full_embedding"] = full_embedding.tobytes().hex()
            else:
                node.embedding = averaged_embedding
        else:
            node.embedding = averaged_embedding

        return node.embedding

    def _to_stored(self, node: TreeNode) -> StoredNode:
        if node.embedding is None:
            raise ValueError(f"Node {node.node_id} is missing an embedding.")

        embedding = node.embedding
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        return StoredNode(
            node_id=node.node_id,
            parent_id=node.parent_id,
            kind=node.kind,
            title=node.title,
            text=node.text,
            metadata=node.metadata,
            embedding=embedding,
            child_ids=[child.node_id for child in node.children],
        )


async def index_and_store(
    text: str,
    *,
    embedding_model: JinaV4Embedder,
    datastore: MilvusPostgresNodeStore,
    document_id: str | None = None,
    title: str = "Untitled",
    metadata: dict | None = None,
    paragraph_embedding_mode: ParagraphEmbeddingMode = "averaged",
) -> list[StoredNode]:
    indexer = RAGIndexer(
        embedding_model=embedding_model,
        datastore=datastore,
        paragraph_embedding_mode=paragraph_embedding_mode,
    )
    return await indexer.index_text(
        text,
        document_id=document_id,
        title=title,
        metadata=metadata,
    )
