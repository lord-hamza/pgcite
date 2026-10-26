from typing import Literal

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    id: str = Field(min_length=1, max_length=200, pattern=r"^[\w.-]+$")
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class IngestRequest(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)


class IngestResponse(BaseModel):
    documents: int
    chunks: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=50)
    mode: Literal["hybrid", "bm25", "vector"] = "hybrid"
    rerank: bool = True


class HitOut(BaseModel):
    chunk_id: int
    doc_id: str
    title: str
    content: str
    score: float


class SearchResponse(BaseModel):
    hits: list[HitOut]
    timings_ms: dict[str, float]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(5, ge=1, le=20)


class CitationOut(BaseModel):
    ref: int
    chunk_id: int
    doc_id: str
    title: str
    snippet: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    model: str
    usage: Usage
    cost_usd: float
    timings_ms: dict[str, float]
