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
    job_id: str | None = None
    status: str = "pending"
    poll_url: str | None = None
    filename: str | None = None
    queue_position: int | None = None
    already_in_rag: bool = False
    message: str | None = None
    document_ids: list[str] = Field(default_factory=list)


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
    use_planner: bool = Field(
        default=True,
        description=(
            "是否用 LLM 将用户问题扩写成多条检索 query。"
            "true：扩写后分别做 Dense+BM25 再融合；"
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


class AnswerRequest(RetrieveRequest):
    """与检索相同入参；生成策略由 .env 的 ANSWER_* 决定。"""


class AnswerResponse(BaseModel):
    question: str
    answer: str
    answer_value: str = ""
    is_blank: bool = False
    ref_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    queries: list[str] = Field(default_factory=list)
    snippets: list[ContextSnippetOut] = Field(default_factory=list)
    retries: int = 0
    ensemble_size: int = 1
