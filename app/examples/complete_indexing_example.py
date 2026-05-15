"""Example: parse → embed → store (same pipeline as /v1/parse)."""

import asyncio

from app.config import (
    JINA_EMBEDDING_DIM,
    JINA_EMBEDDING_TASK,
    JINA_PARAGRAPH_MODE,
    MILVUS_INDEX_TYPE,
    MILVUS_METRIC,
    RAG_TABLE_PREFIX,
    resolve_jina_model_path,
)
from app.document_parser import (
    JinaV4Embedder,
    MilvusPostgresNodeStore,
    RAGIndexer,
    text_to_payload,
)


async def main() -> None:
    embedder = JinaV4Embedder(
        model_name=resolve_jina_model_path(),
        task=JINA_EMBEDDING_TASK,
        truncate_dim=JINA_EMBEDDING_DIM,
    )
    datastore = MilvusPostgresNodeStore(
        dimensions=embedder.dimension,
        table_prefix=RAG_TABLE_PREFIX,
        paragraph_search_mode=JINA_PARAGRAPH_MODE,
        index_type=MILVUS_INDEX_TYPE,
        metric=MILVUS_METRIC,
    )
    indexer = RAGIndexer(
        embedding_model=embedder,
        datastore=datastore,
        paragraph_embedding_mode=JINA_PARAGRAPH_MODE,
    )

    payload = text_to_payload(
        document_id="example-doc",
        title="Example",
        text="Python is a programming language. It is widely used for machine learning.",
        metadata={"source": "example"},
    )
    nodes = await indexer.index_document(payload)
    print(f"Indexed {len(nodes)} nodes for {payload.document_id}")

    query_vec = (await embedder.embed_text(["What is Python?"]))[0]
    hits = await datastore.search(query_vec, k=3)
    for i, match in enumerate(hits, 1):
        print(f"{i}. [{match.node.kind.value}] {match.score:.4f} — {match.node.text[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
