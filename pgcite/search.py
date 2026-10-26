from dataclasses import dataclass, replace

import numpy as np
import psycopg

# BM25 with Lucene-style IDF. The @@ filter uses the GIN index with OR semantics, and doc
# frequencies stay exact because every chunk holding any query lexeme ends up in `matches`.
# Equal scores are common, so ties are ordered by doc_id and position instead of row order.
BM25_SQL = """
WITH query_terms AS (
    SELECT DISTINCT lexeme FROM unnest(to_tsvector('english', %(query)s))
),
corpus AS (
    SELECT count(*)::float8 AS n, avg(token_count)::float8 AS avgdl FROM chunks
),
matches AS (
    SELECT c.id, c.token_count, t.lexeme, cardinality(t.positions) AS tf
    FROM chunks c
    CROSS JOIN LATERAL unnest(c.tsv) AS t
    JOIN query_terms q ON q.lexeme = t.lexeme
    WHERE c.tsv @@ replace(plainto_tsquery('english', %(query)s)::text, ' & ', ' | ')::tsquery
),
doc_freq AS (
    SELECT lexeme, count(*)::float8 AS df FROM matches GROUP BY lexeme
),
scores AS (
    SELECT m.id,
           sum(ln(1 + (corpus.n - d.df + 0.5) / (d.df + 0.5))
               * m.tf * (%(k1)s + 1)
               / (m.tf + %(k1)s * (1 - %(b)s + %(b)s * m.token_count / corpus.avgdl))) AS score
    FROM matches m
    JOIN doc_freq d USING (lexeme)
    CROSS JOIN corpus
    GROUP BY m.id
)
SELECT c.id, c.doc_id, c.title, c.content, s.score
FROM scores s JOIN chunks c ON c.id = s.id
ORDER BY s.score DESC, c.doc_id, c.position
LIMIT %(limit)s
"""

VECTOR_SQL = """
SELECT id, doc_id, title, content, 1 - (embedding <=> %(embedding)s) AS score
FROM chunks
ORDER BY embedding <=> %(embedding)s
LIMIT %(limit)s
"""


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    title: str
    content: str
    score: float


def _hits(rows: list[tuple]) -> list[Hit]:
    return [Hit(row[0], row[1], row[2], row[3], float(row[4])) for row in rows]


def bm25_search(
    conn: psycopg.Connection, query: str, limit: int, k1: float = 1.2, b: float = 0.75
) -> list[Hit]:
    params = {"query": query, "limit": limit, "k1": k1, "b": b}
    return _hits(conn.execute(BM25_SQL, params).fetchall())


def vector_search(conn: psycopg.Connection, embedding: np.ndarray, limit: int) -> list[Hit]:
    return _hits(conn.execute(VECTOR_SQL, {"embedding": embedding, "limit": limit}).fetchall())


def rrf_fuse(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[int, float] = {}
    hits: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (k + rank)
            hits.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [replace(hits[chunk_id], score=scores[chunk_id]) for chunk_id in ordered]
