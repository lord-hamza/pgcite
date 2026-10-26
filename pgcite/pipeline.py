import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Literal

from psycopg_pool import ConnectionPool

from pgcite.chunking import Document, chunk_text
from pgcite.config import Settings, get_settings
from pgcite.db import create_pool
from pgcite.generate import Answer, Generator, build_generator
from pgcite.models import Embedder, Reranker, load_models
from pgcite.search import Hit, bm25_search, rrf_fuse, vector_search

Mode = Literal["hybrid", "bm25", "vector"]

INSERT_SQL = """
INSERT INTO chunks (doc_id, title, position, content, token_count, embedding)
VALUES (%s, %s, %s, %s, %s, %s)
"""


class StageTimer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = round((time.perf_counter() - start) * 1000, 2)


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]


@dataclass
class AskResult:
    answer: Answer
    hits: list[Hit]
    timings_ms: dict[str, float]


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        pool: ConnectionPool,
        embedder: Embedder,
        reranker: Reranker,
        generator: Generator,
    ):
        self.settings = settings
        self.pool = pool
        self.embedder = embedder
        self.reranker = reranker
        self.generator = generator

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "RAGPipeline":
        settings = settings or get_settings()
        pool = create_pool(settings.database_url, settings.embedding_dim)
        embedder, reranker = load_models(settings)
        return cls(settings, pool, embedder, reranker, build_generator(settings, embedder))

    def close(self) -> None:
        self.pool.close()

    def __enter__(self) -> "RAGPipeline":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def ingest(self, documents: list[Document]) -> int:
        chunks = [
            (document, position, text)
            for document in documents
            for position, text in enumerate(chunk_text(document.text, self.settings.chunk_words))
        ]
        rows = []
        if chunks:
            embeddings = self.embedder.embed_passages([f"{d.title}\n{t}" for d, _, t in chunks])
            for (document, position, text), embedding in zip(chunks, embeddings, strict=True):
                token_count = len(document.title.split()) + len(text.split())
                rows.append((document.id, document.title, position, text, token_count, embedding))
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("DELETE FROM chunks WHERE doc_id = ANY(%s)", ([d.id for d in documents],))
            with conn.cursor() as cursor:
                cursor.executemany(INSERT_SQL, rows)
        return len(rows)

    def delete(self, doc_id: str) -> int:
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,)).rowcount

    def search(
        self, query: str, k: int | None = None, mode: Mode = "hybrid", rerank: bool = True
    ) -> SearchResult:
        timer = StageTimer()
        with timer.stage("total"):
            hits = self._retrieve(query, k or self.settings.top_k, mode, rerank, timer)
        return SearchResult(hits, timer.ms)

    def ask(
        self, question: str, k: int | None = None, mode: Mode = "hybrid", rerank: bool = True
    ) -> AskResult:
        timer = StageTimer()
        with timer.stage("total"):
            hits = self._retrieve(question, k or self.settings.top_k, mode, rerank, timer)
            with timer.stage("generate"):
                answer = self.generator.generate(question, hits)
        return AskResult(answer, hits, timer.ms)

    def _retrieve(
        self, query: str, k: int, mode: Mode, rerank: bool, timer: StageTimer
    ) -> list[Hit]:
        limit = self.settings.candidates
        rankings = []
        if mode != "bm25":
            with timer.stage("embed"):
                query_embedding = self.embedder.embed_query(query)
        with self.pool.connection() as conn:
            if mode != "vector":
                with timer.stage("bm25"):
                    rankings.append(bm25_search(conn, query, limit))
            if mode != "bm25":
                with timer.stage("vector"):
                    rankings.append(vector_search(conn, query_embedding, limit))
        if mode == "hybrid":
            with timer.stage("fuse"):
                hits = rrf_fuse(rankings)
        else:
            hits = rankings[0]
        if rerank and hits:
            with timer.stage("rerank"):
                candidates = hits[: max(k, self.settings.rerank_candidates)]
                passages = [f"{hit.title}\n{hit.content}" for hit in candidates]
                scores = self.reranker.score(query, passages)
                reranked = [
                    replace(hit, score=score) for hit, score in zip(candidates, scores, strict=True)
                ]
                hits = sorted(reranked, key=lambda hit: hit.score, reverse=True)
        return hits[:k]
