import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    position INT NOT NULL,
    content TEXT NOT NULL,
    token_count INT NOT NULL,
    tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', title || ' ' || content)) STORED,
    embedding VECTOR({dim}) NOT NULL,
    UNIQUE (doc_id, position)
);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS query_log (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    endpoint TEXT NOT NULL,
    query TEXT NOT NULL,
    model TEXT,
    total_ms REAL NOT NULL,
    timings JSONB NOT NULL,
    input_tokens INT NOT NULL DEFAULT 0,
    output_tokens INT NOT NULL DEFAULT 0,
    cost_usd NUMERIC(12, 6) NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS query_log_created_at_idx ON query_log (created_at);
"""

SUMMARY_SQL = """
SELECT count(*),
       percentile_cont(0.5) WITHIN GROUP (ORDER BY total_ms),
       percentile_cont(0.95) WITHIN GROUP (ORDER BY total_ms),
       percentile_cont(0.99) WITHIN GROUP (ORDER BY total_ms),
       avg(cost_usd),
       sum(cost_usd)
FROM query_log
WHERE created_at > now() - make_interval(hours => %(hours)s)
"""

STAGES_SQL = """
SELECT stage.key,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY stage.value::float8),
       percentile_cont(0.95) WITHIN GROUP (ORDER BY stage.value::float8)
FROM query_log, jsonb_each_text(timings) AS stage
WHERE created_at > now() - make_interval(hours => %(hours)s) AND stage.key <> 'total'
GROUP BY stage.key
ORDER BY stage.key
"""


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    # The default ef_search of 40 would cap vector results below `candidates`.
    conn.execute("SET hnsw.ef_search = 100")


def create_pool(database_url: str, dim: int) -> ConnectionPool:
    # The extension has to exist before register_vector can look up the type, and the advisory
    # lock stops several API workers booting at once from racing on the DDL.
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(20240501)")
        conn.execute(SCHEMA.format(dim=dim))
    return ConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True},
        configure=_configure,
        open=True,
    )


def log_query(
    pool: ConnectionPool,
    *,
    endpoint: str,
    query: str,
    model: str | None,
    timings: dict[str, float],
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO query_log (endpoint, query, model, total_ms, timings,"
            " input_tokens, output_tokens, cost_usd) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (
                endpoint,
                query,
                model,
                timings["total"],
                Jsonb(timings),
                input_tokens,
                output_tokens,
                cost_usd,
            ),
        )


def _round(value, digits: int = 2) -> float:
    return round(float(value or 0), digits)


def query_stats(pool: ConnectionPool, hours: int = 24) -> dict:
    with pool.connection() as conn:
        count, p50, p95, p99, avg_cost, total_cost = conn.execute(
            SUMMARY_SQL, {"hours": hours}
        ).fetchone()
        stages = conn.execute(STAGES_SQL, {"hours": hours}).fetchall()
    return {
        "window_hours": hours,
        "queries": count,
        "latency_ms": {"p50": _round(p50), "p95": _round(p95), "p99": _round(p99)},
        "stages": {name: {"p50": _round(s50), "p95": _round(s95)} for name, s50, s95 in stages},
        "cost_usd": {"avg": _round(avg_cost, 6), "total": _round(total_cost, 6)},
    }
