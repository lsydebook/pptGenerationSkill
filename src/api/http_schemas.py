from typing import Any

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    text: str
    metadata: dict[str, Any]


class IndexingSummary(BaseModel):
    document_id: str
    nodes_indexed: int


class ParseResponse(BaseModel):
    filename: str
    content_type: str | None
    documents: list[ParsedDocument]
    indexing: list[IndexingSummary]


class ParseJobAccepted(BaseModel):
    job_id: str
    status: str = "pending"
    poll_url: str
    filename: str | None = None
    queue_position: int | None = None


class JobResultSummary(BaseModel):
    filename: str
    content_type: str | None = None
    indexing: list[IndexingSummary]


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    queue_position: int | None = None
    error: str | None = None
    result: JobResultSummary | None = None


class RetrieveRequest(BaseModel):
    question: str
    top_k: int | None = Field(
        default=None,
        description=(
            "每条检索 query 的 Dense 向量召回数量；不传则用 .env 的 RETRIEVAL_TOP_K（默认 8）"
        ),
    )
    bm25_top_k: int | None = Field(
        default=None,
        description=(
            "每条检索 query 的 BM25 关键词召回数量；0 表示关闭 BM25；"
            "不传则用 .env 的 RETRIEVAL_BM25_TOP_K（默认 4）"
        ),
    )
    use_planner: bool = Field(
        default=True,
        description=(
            "是否用 LLM 将用户问题扩写成多条检索 query。"
            "true：扩写后分别做 Dense+BM25 再合并重排；"
            "false：仅用原始 question 检索一次"
        ),
    )


class RetrievalMatchOut(BaseModel):
    node_id: str
    kind: str
    text: str
    score: float
    metadata: dict[str, Any]


class ContextSnippetOut(BaseModel):
    node_id: str
    document_title: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any]


class RetrieveResponse(BaseModel):
    question: str
    queries: list[str]
    matches: list[RetrievalMatchOut]
    snippets: list[ContextSnippetOut]
