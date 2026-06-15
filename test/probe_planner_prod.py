"""Probe planner via production init path."""
import asyncio
import time

import src.config.env_loader  # noqa: F401
import src.rag_retrieval as rag_retrieval


async def main() -> None:
    await rag_retrieval.init_retrieval()
    assert rag_retrieval._planner is not None
    question = "查询一下青蒿素相关知识"
    t0 = time.perf_counter()
    queries = await rag_retrieval._planner.plan(question)
    print(f"elapsed={time.perf_counter() - t0:.2f}s queries={queries}")
    rag_retrieval.shutdown_retrieval()


if __name__ == "__main__":
    asyncio.run(main())
