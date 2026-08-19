"""Evaluate this RAG service on CRUD-RAG Read (Chinese news QA).

Flow: load split_merged.json → ingest gold news via POST /v1/parse →
POST /v1/retrieve + /v1/answer → Hit@k + char-F1 + optional RAGAS.

Corpus is sampled gold news plus optional unused 1-doc news as distractors
(not the 80k dump). Use --from-results to rescore an existing JSON with RAGAS.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT = ROOT / "data" / "eval" / "split_merged.json"
RESULTS_DIR = ROOT / "eval" / "results"
MANIFEST_PATH = ROOT / "data" / "eval" / "ingest_manifest.json"
QA_TASKS = ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs")

load_dotenv(ROOT / ".env")


def _patch_ragas_vertexai() -> None:
    """Installed ragas imports VertexAI from langchain_community; stub it."""
    vertexai = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # noqa: D401
        pass

    vertexai.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules.setdefault("langchain_community.chat_models.vertexai", vertexai)
    import langchain_community.llms as llms_mod

    if not hasattr(llms_mod, "VertexAI"):

        class VertexAI:  # noqa: D401
            pass

        llms_mod.VertexAI = VertexAI  # type: ignore[attr-defined]


def content_hash12(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def document_id_for_bytes(data: bytes, filename: str) -> str:
    safe = os.path.basename(filename).strip() or "upload"
    stem = os.path.splitext(safe)[0] or safe
    return f"{stem}-{content_hash12(data)}"


def news_filename(text: str) -> str:
    digest = content_hash12(text.encode("utf-8"))
    return f"crud_{digest}.txt"


def extract_news(item: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for key, value in item.items():
        if "news" not in str(key).lower():
            continue
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        elif isinstance(value, list):
            texts.extend(str(part).strip() for part in value if str(part).strip())
    return texts


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(str(part) for part in value if part is not None)
    return str(value)


def load_questions(split_path: Path, tasks: list[str], limit: int) -> list[dict[str, Any]]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    per_task = max(1, math.ceil(limit / max(len(tasks), 1)))
    for task in tasks:
        items = data.get(task) or []
        take = items[:per_task]
        for index, item in enumerate(take):
            question = as_text(item.get("questions")).strip()
            answer = as_text(item.get("answers")).strip()
            news = extract_news(item)
            if not question or not answer or not news:
                continue
            rows.append(
                {
                    "id": str(item.get("ID") or f"{task}-{index}"),
                    "task": task,
                    "question": question,
                    "reference": answer,
                    "news": news,
                }
            )
            if len(rows) >= limit:
                return rows
    return rows[:limit]


def load_extra_news(split_path: Path, used_texts: set[str], extra: int) -> list[str]:
    """Unused 1-doc news texts used as distractors in the same collection."""
    if extra <= 0:
        return []
    data = json.loads(split_path.read_text(encoding="utf-8"))
    extras: list[str] = []
    seen: set[str] = set(used_texts)
    for item in data.get("questanswer_1doc") or []:
        for text in extract_news(item):
            if text in seen:
                continue
            seen.add(text)
            extras.append(text)
            if len(extras) >= extra:
                return extras
    return extras


def char_f1(pred: str, gold: str) -> float:
    pred_chars = [ch for ch in pred if not ch.isspace()]
    gold_chars = [ch for ch in gold if not ch.isspace()]
    if not pred_chars and not gold_chars:
        return 1.0
    if not pred_chars or not gold_chars:
        return 0.0
    overlap = sum((Counter(pred_chars) & Counter(gold_chars)).values())
    precision = overlap / len(pred_chars)
    recall = overlap / len(gold_chars)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    return 1.0 if top & set(relevant_ids) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def snippet_doc_ids(payload: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for snippet in payload.get("snippets") or []:
        meta = snippet.get("metadata") or {}
        doc_id = str(meta.get("document_id") or "").strip()
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        ids.append(doc_id)
    return ids


def snippet_texts(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for snippet in payload.get("snippets") or []:
        text = str(snippet.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts


class RagClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout_s, connect=30.0)

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    async def parse_text(self, text: str, filename: str, note: str) -> tuple[int, dict[str, Any]]:
        data = {"note": note}
        files = {"file": (filename, text.encode("utf-8"), "text/plain")}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/parse",
                data=data,
                files=files,
            )
            if response.status_code not in {200, 202}:
                response.raise_for_status()
            return response.status_code, response.json()

    async def wait_job(self, job_id: str, poll_s: float = 1.5) -> dict[str, Any]:
        deadline = time.time() + 600
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while time.time() < deadline:
                response = await client.get(f"{self.base_url}/v1/jobs/{job_id}")
                response.raise_for_status()
                payload = response.json()
                status = payload.get("status")
                if status in {"succeeded", "failed"}:
                    return payload
                await asyncio.sleep(poll_s)
        raise TimeoutError(f"job {job_id} timed out")

    async def retrieve(self, question: str, use_planner: bool) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/retrieve",
                json={"question": question, "use_planner": use_planner},
            )
            response.raise_for_status()
            return response.json()

    async def answer(self, question: str, use_planner: bool) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/answer",
                json={"question": question, "use_planner": use_planner},
            )
            response.raise_for_status()
            return response.json()


async def ingest_news(
    client: RagClient,
    news_texts: list[str],
    *,
    concurrency: int,
) -> dict[str, str]:
    """Return mapping news_text -> document_id."""
    unique: list[str] = []
    seen: set[str] = set()
    for text in news_texts:
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)

    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {}

    sem = asyncio.Semaphore(concurrency)
    mapping: dict[str, str] = {}
    lock = asyncio.Lock()
    done = 0

    async def one(text: str) -> tuple[str, str]:
        nonlocal done
        filename = news_filename(text)
        doc_id = document_id_for_bytes(text.encode("utf-8"), filename)
        async with lock:
            if manifest.get(doc_id):
                done += 1
                print(f"ingest {done}/{len(unique)} skip {doc_id}", flush=True)
                return text, doc_id
        async with sem:
            status, payload = await client.parse_text(
                text,
                filename,
                note="crud-rag-eval",
            )
            if status == 200 and payload.get("already_in_rag"):
                ids = payload.get("document_ids") or [doc_id]
                resolved = ids[0]
            else:
                job_id = payload.get("job_id")
                if not job_id:
                    raise RuntimeError(f"parse missing job_id: {payload}")
                job = await client.wait_job(job_id)
                if job.get("status") != "succeeded":
                    raise RuntimeError(f"ingest failed {filename}: {job.get('error')}")
                indexing = (job.get("result") or {}).get("indexing") or []
                resolved = indexing[0]["document_id"] if indexing else doc_id
        async with lock:
            done += 1
            manifest[resolved] = filename
            MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST_PATH.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"ingest {done}/{len(unique)} -> {resolved}", flush=True)
        return text, resolved

    pairs = await asyncio.gather(*[one(text) for text in unique])
    for news, doc_id in pairs:
        mapping[news] = doc_id
    return mapping


def mean(values: list[float]) -> float:
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


async def score_ragas(
    rows: list[dict[str, Any]],
) -> None:
    _patch_ragas_vertexai()
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

    api_key = os.getenv("PLANNER_API_KEY") or os.getenv("EMBEDDING_API_KEY")
    base_url = os.getenv("PLANNER_BASE_URL") or os.getenv("EMBEDDING_BASE_URL")
    model = os.getenv("PLANNER_MODEL", "qwen-latest")
    if not api_key or not base_url:
        raise RuntimeError("PLANNER_API_KEY / PLANNER_BASE_URL missing in .env")

    llm_kwargs: dict[str, Any] = {"max_tokens": 8192}
    # qwen-latest thinking tokens eat the completion budget; match the service planner.
    thinking = os.getenv("PLANNER_ENABLE_THINKING", "").strip().lower()
    if thinking not in {"1", "true", "yes", "on"}:
        llm_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False},
        }
    llm = llm_factory(
        model,
        provider="openai",
        client=AsyncOpenAI(api_key=api_key, base_url=base_url),
        **llm_kwargs,
    )
    faithfulness = Faithfulness(llm=llm)
    precision = ContextPrecision(llm=llm)
    recall = ContextRecall(llm=llm)

    for index, row in enumerate(rows, start=1):
        raw_contexts = row.get("retrieved_contexts") or []
        contexts = [text[:1800] for text in raw_contexts[:6] if text]
        answer = row.get("pred_answer") or ""
        if not contexts or not answer:
            row["faithfulness"] = float("nan")
            row["context_precision"] = float("nan")
            row["context_recall"] = float("nan")
            continue
        try:
            faith = await faithfulness.ascore(
                user_input=row["question"],
                response=answer,
                retrieved_contexts=contexts,
            )
            prec = await precision.ascore(
                user_input=row["question"],
                reference=row["reference"],
                retrieved_contexts=contexts,
            )
            rec = await recall.ascore(
                user_input=row["question"],
                retrieved_contexts=contexts,
                reference=row["reference"],
            )
            row["faithfulness"] = float(faith.value)
            row["context_precision"] = float(prec.value)
            row["context_recall"] = float(rec.value)
        except Exception as exc:  # noqa: BLE001
            print(f"ragas failed id={row['id']}: {exc}")
            row["faithfulness"] = float("nan")
            row["context_precision"] = float("nan")
            row["context_recall"] = float("nan")
        print(
            f"ragas {index}/{len(rows)} faith={row['faithfulness']:.3f} "
            f"p={row['context_precision']:.3f} r={row['context_recall']:.3f}",
            flush=True,
        )


def build_summary(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    corpus_docs: int | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "dataset": "CRUD-RAG Read (split_merged.json)",
        "n": len(records),
        "tasks": args.tasks,
        "use_planner": args.use_planner,
        "hit@5": mean([float(r["hit@5"]) for r in records]),
        "hit@10": mean([float(r["hit@10"]) for r in records]),
        "mrr": mean([float(r["mrr"]) for r in records]),
        "char_f1": mean([float(r.get("char_f1", math.nan)) for r in records]),
        "blank_rate": mean([1.0 if r.get("is_blank") else 0.0 for r in records]),
    }
    if corpus_docs is not None:
        summary["corpus_docs"] = corpus_docs
    if any("faithfulness" in r for r in records):
        summary["faithfulness"] = mean(
            [float(r.get("faithfulness", math.nan)) for r in records]
        )
        summary["context_precision"] = mean(
            [float(r.get("context_precision", math.nan)) for r in records]
        )
        summary["context_recall"] = mean(
            [float(r.get("context_recall", math.nan)) for r in records]
        )
    return summary


def write_output(summary: dict[str, Any], records: list[dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_json = RESULTS_DIR / f"crud_read_{stamp}.json"
    payload = {"summary": summary, "rows": records}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== CRUD-RAG Read summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"wrote {out_json}")
    return out_json


async def run(args: argparse.Namespace) -> int:
    if args.from_results:
        payload = json.loads(args.from_results.read_text(encoding="utf-8"))
        records = payload.get("rows") or []
        if not records:
            print("empty --from-results file", file=sys.stderr)
            return 1
        print(f"rescoring {len(records)} rows from {args.from_results}")
        await score_ragas(records)
        summary = dict(payload.get("summary") or {})
        summary.update(build_summary(records, args))
        write_output(summary, records)
        return 0

    if not args.split.exists():
        print(
            f"missing {args.split}. Run: uv run python eval/download_crud_rag.py",
            file=sys.stderr,
        )
        return 1

    questions = load_questions(args.split, args.tasks, args.limit)
    if not questions:
        print("no usable QA rows (need questions/answers/news*)", file=sys.stderr)
        return 1

    client = RagClient(args.base_url, timeout_s=args.timeout)
    health = await client.health()
    print(f"health={health}")

    news_texts = [text for row in questions for text in row["news"]]
    used_texts = set(news_texts)
    news_texts.extend(load_extra_news(args.split, used_texts, args.extra_docs))
    unique_news = list(dict.fromkeys(news_texts))
    if args.skip_ingest:
        mapping = {
            text: document_id_for_bytes(text.encode("utf-8"), news_filename(text))
            for text in unique_news
        }
    else:
        print(f"ingesting {len(unique_news)} unique news docs")
        mapping = await ingest_news(client, unique_news, concurrency=args.ingest_concurrency)

    records: list[dict[str, Any]] = []
    for index, row in enumerate(questions, start=1):
        relevant = [mapping[text] for text in row["news"] if text in mapping]
        print(f"[{index}/{len(questions)}] retrieve {row['id']}", flush=True)
        retrieved = await client.retrieve(row["question"], args.use_planner)
        retrieved_ids = snippet_doc_ids(retrieved)
        contexts = snippet_texts(retrieved)
        if args.skip_answer:
            pred = ""
            is_blank = False
        else:
            print(f"[{index}/{len(questions)}] answer {row['id']}", flush=True)
            answered = await client.answer(row["question"], args.use_planner)
            pred = str(answered.get("answer") or "")
            is_blank = bool(answered.get("is_blank"))
        record = {
            "id": row["id"],
            "task": row["task"],
            "question": row["question"],
            "reference": row["reference"],
            "relevant_doc_ids": relevant,
            "retrieved_doc_ids": retrieved_ids,
            "retrieved_contexts": contexts,
            "pred_answer": pred,
            "is_blank": is_blank,
            "hit@5": hit_at_k(retrieved_ids, relevant, 5),
            "hit@10": hit_at_k(retrieved_ids, relevant, 10),
            "mrr": mrr(retrieved_ids, relevant),
            "char_f1": char_f1(pred, row["reference"]) if pred else float("nan"),
        }
        records.append(record)

    if not args.skip_ragas and not args.skip_answer:
        print("scoring RAGAS (judge = PLANNER_MODEL)")
        await score_ragas(records)

    summary = build_summary(records, args, corpus_docs=len(unique_news))
    write_output(summary, records)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--base-url", default=os.getenv("EVAL_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["questanswer_1doc"],
        choices=list(QA_TASKS),
    )
    parser.add_argument("--use-planner", dest="use_planner", action="store_true", default=True)
    parser.add_argument("--no-planner", dest="use_planner", action="store_false")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument(
        "--extra-docs",
        type=int,
        default=0,
        help="ingest unused 1-doc news as distractors so Hit@k is not on a tiny gold-only pool",
    )
    parser.add_argument(
        "--from-results",
        type=Path,
        help="rescore an existing results JSON with RAGAS (no ingest/retrieve/answer)",
    )
    parser.add_argument("--ingest-concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
