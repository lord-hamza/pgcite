import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request

from pgcite.chunking import Document
from pgcite.db import log_query, query_stats
from pgcite.pipeline import RAGPipeline
from pgcite.schemas import (
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    Usage,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = RAGPipeline.from_settings()
    yield
    app.state.pipeline.close()


app = FastAPI(title="pgcite", version="0.1.0", lifespan=lifespan)


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


Pipeline = Annotated[RAGPipeline, Depends(get_pipeline)]


def require_api_key(pipeline: Pipeline, x_api_key: Annotated[str | None, Header()] = None) -> None:
    expected = pipeline.settings.api_key
    if expected and not secrets.compare_digest((x_api_key or "").encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


protected = [Depends(require_api_key)]


@app.get("/health")
def health(pipeline: Pipeline) -> dict:
    with pipeline.pool.connection() as conn:
        chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    return {"status": "ok", "chunks": chunks, "generator": pipeline.generator.name}


@app.post("/documents", dependencies=protected)
def add_documents(body: IngestRequest, pipeline: Pipeline) -> IngestResponse:
    documents = [Document(doc.id, doc.title, doc.text) for doc in body.documents]
    return IngestResponse(documents=len(documents), chunks=pipeline.ingest(documents))


@app.delete("/documents/{doc_id}", dependencies=protected)
def delete_document(doc_id: str, pipeline: Pipeline) -> dict[str, int]:
    deleted = pipeline.delete(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No document with id {doc_id!r}")
    return {"deleted_chunks": deleted}


@app.post("/search", dependencies=protected)
def search(body: SearchRequest, pipeline: Pipeline) -> SearchResponse:
    result = pipeline.search(body.query, k=body.k, mode=body.mode, rerank=body.rerank)
    log_query(
        pipeline.pool, endpoint="search", query=body.query, model=None, timings=result.timings_ms
    )
    return SearchResponse(hits=[asdict(hit) for hit in result.hits], timings_ms=result.timings_ms)


@app.post("/ask", dependencies=protected)
def ask(body: AskRequest, pipeline: Pipeline) -> AskResponse:
    try:
        result = pipeline.ask(body.question, k=body.k)
    except anthropic.APIError as error:
        raise HTTPException(status_code=502, detail=f"Generation failed: {error}") from error
    answer = result.answer
    log_query(
        pipeline.pool,
        endpoint="ask",
        query=body.question,
        model=answer.model,
        timings=result.timings_ms,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=answer.cost_usd,
    )
    return AskResponse(
        answer=answer.text,
        citations=[asdict(citation) for citation in answer.citations],
        model=answer.model,
        usage=Usage(input_tokens=answer.input_tokens, output_tokens=answer.output_tokens),
        cost_usd=answer.cost_usd,
        timings_ms=result.timings_ms,
    )


@app.get("/stats")
def stats(pipeline: Pipeline, hours: Annotated[int, Query(ge=1, le=720)] = 24) -> dict:
    return query_stats(pipeline.pool, hours)
