"""Smoke test for RAG retrieval pipeline."""

from __future__ import annotations

import asyncio

from src.config.logging_config import setup_logging
from src.rag_parsing import init_parsing, shutdown_parsing

setup_logging()
from src.rag_retrieval import RetrievalRequest, init_retrieval, run_retrieval, shutdown_retrieval


async def main() -> None:
    await init_parsing()
    await init_retrieval()
    result = await run_retrieval(
        RetrievalRequest(question="本周实验进展如何？", use_planner=True)
    )
    print("queries:", result.queries)
    print("matches:", len(result.matches))
    print("snippets:", len(result.snippets))
    if result.snippets:
        print("first snippet:", result.snippets[0]["node_id"])
    shutdown_retrieval()
    shutdown_parsing()


if __name__ == "__main__":
    asyncio.run(main())
