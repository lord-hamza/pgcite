# pgcite

pgcite answers questions about your documents and shows exactly where each answer came from. It runs on plain PostgreSQL with pgvector, mixes keyword and semantic search, reranks what it finds and checks its own answer quality on every push.

| part | what pgcite uses |
|---|---|
| Search | BM25 written in SQL plus pgvector HNSW, merged with Reciprocal Rank Fusion |
| Reranking | a local cross-encoder (ONNX, runs on CPU) |
| Answers | inline `[n]` citations from an offline extractive generator or an LLM through the Anthropic API |
| Quality gate | 40 golden questions and 5 metrics, CI fails when any of them drops |
| Tracking | latency per stage, tokens and cost on every response, stored in Postgres |
| Offline | no API keys needed once two small models are downloaded |

## pgcite explained to a 5-year-old

Imagine a big library. You walk in with a question: "How long does a lost toy stay in the lost and found box?"

pgcite sends two helpers to look. One looks for pages with the exact words you used. The other looks for pages that mean the same thing, even when the words are different. Both bring back their favorite pages and put them in one pile.

Then a very careful reader goes through the pile and keeps the five pages that really answer your question. pgcite gives you the answer and points at the page it came from: "It says so right here, on page 2."

Every time someone changes pgcite, a teacher gives it the same 40-question quiz. If it scores worse than last time, the teacher says "not yet" and the change waits until it's fixed.

What each part really is:

| in the story | in pgcite | why it helps |
|---|---|---|
| the exact-words helper | BM25 keyword search in SQL | catches codes and names like `FH-4031` |
| the same-meaning helper | vector search with pgvector | matches "get my money back" to a page about refunds |
| one pile from two | Reciprocal Rank Fusion | combines both lists without tuning any weights |
| the careful reader | cross-encoder reranker | reads the question and the page together |
| pointing at the page | `[n]` citations | every sentence can be checked |
| the quiz | golden-set eval in CI | quality can't quietly get worse |

## How it works

### Answering a question

```mermaid
flowchart LR
    Q([question]) --> E[embed query<br/>bge-small-en-v1.5]
    Q --> B[BM25 in SQL<br/>tsvector + GIN]
    E --> V[vector search<br/>pgvector HNSW]
    B --> F[RRF fusion]
    V --> F
    F --> R[cross-encoder rerank<br/>top 20]
    R --> K[top 5 chunks]
    K --> G{generator}
    G -->|extractive| A([answer with citations])
    G -->|LLM| A
    A -.-> L[(query_log)]
    L -.-> S[GET /stats]
```

**BM25 in SQL.** Postgres' `ts_rank` isn't BM25, so `pgcite/search.py` computes real BM25 (k1 1.2, b 0.75) in one query. The GIN index finds chunks with any query term. Term counts come from the stored `tsvector` and document frequencies are counted over the matches. No extension needed.

**Fusion and reranking.** BM25 scores and cosine similarities sit on different scales, so Reciprocal Rank Fusion merges them by rank alone (`1 / (60 + rank)`). The top 20 then go through a cross-encoder that reads the question and each chunk together. The best five become the context.

**Citations.** Passages are numbered `[1]` to `[5]` and each answer sentence cites its source (`... within 30 days [2].`). References to passages that don't exist get dropped.

**Generators.** `extractive` (the default) picks the context sentences closest to the question. It's deterministic and free, which keeps the demo and CI offline. `anthropic` sends the passages to an LLM (`claude-opus-5-5` by default, low effort) and asks it to answer only from them.

### Indexing and the quality gate

```mermaid
flowchart LR
    subgraph indexing [Indexing]
        D[markdown files] --> C[chunks of about 120 words]
        C --> M[embed passages]
        M --> T[(chunks table<br/>tsvector + vector)]
    end
    subgraph gate [CI quality gate]
        G[40 golden questions] --> R[run the full pipeline]
        R --> S[score 5 metrics]
        S --> X{more than 0.02<br/>below baseline?}
        X -->|yes| FAIL[build fails]
        X -->|no| PASS[report in job summary]
    end
    T --> R
```

Re-indexing a document replaces its chunks in one transaction, so `make ingest` is safe to repeat.

## Quickstart

### With Docker

```bash
make demo
```

It builds the image, starts Postgres and the API, indexes `data/docs` and asks a sample question. By hand:

```bash
docker compose up -d --build
docker compose exec api python scripts/ingest.py
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How long are deleted pages kept in the trash?"}'
```

### Without Docker

You need Postgres 17 with pgvector (`brew install postgresql@17 pgvector` or `apt install postgresql-17 postgresql-17-pgvector`):

```bash
createuser rag --pwprompt                  # password: rag
createdb rag --owner rag
psql -d rag -c 'CREATE EXTENSION vector'   # as a superuser

python3 -m venv .venv && source .venv/bin/activate
make install
export DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
make ingest
make serve                                 # http://localhost:8000/docs
```

Or from Python:

```bash
python examples/quickstart.py "Which plans include SAML single sign-on?"
```

The first run downloads two models (about 150 MB) into `.cache/models`. After that it works offline. For LLM answers set `GENERATOR=anthropic` and `ANTHROPIC_API_KEY`.

## API

| method | path | what it does |
|---|---|---|
| GET | `/health` | database check, chunk count, active generator |
| POST | `/documents` | chunks, embeds and stores documents |
| DELETE | `/documents/{doc_id}` | removes a document |
| POST | `/search` | `hybrid`, `bm25` or `vector` search with reranking on or off |
| POST | `/ask` | cited answer with timings, token usage and cost |
| GET | `/stats?hours=24` | latency percentiles and spend from `query_log` |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What is the default value of sync.batch_size?"}'
```

```json
{
  "answer": "The default `sync.batch_size` is 200 and the maximum is 500 [1]. `--batch-size` overrides the `sync.batch_size` setting for one run [2].",
  "citations": [
    {
      "ref": 1,
      "chunk_id": 354,
      "doc_id": "sync-troubleshooting",
      "title": "Sync troubleshooting",
      "snippet": "The `sync.batch_size` setting controls how many changes go into each sync request.\nThe default `sync.batch_size` is 200 and the maximum is 500.\nOn slow or unstable connections, lower it to 50 so each ..."
    },
    {
      "ref": 2,
      "chunk_id": 293,
      "doc_id": "cli-reference",
      "title": "CLI reference",
      "snippet": "It asks for confirmation unless you also pass `--yes`.\n`--batch-size` overrides the `sync.batch_size` setting for one run.\nRate limits and exit codes\nWhen the API rejects a request for exceeding the ..."
    }
  ],
  "model": "extractive",
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0
  },
  "cost_usd": 0.0,
  "timings_ms": {
    "embed": 46.73,
    "bm25": 36.22,
    "vector": 3.3,
    "fuse": 0.11,
    "rerank": 587.8,
    "generate": 297.09,
    "total": 971.44
  }
}
```

## Evaluation

`evals/golden.jsonl` has 40 questions about `data/docs`, 25 help-center articles for a made-up product. The articles repeat similar facts on purpose (limits per plan, neighbouring error codes) so sloppy retrieval grabs the wrong one. Questions mix exact identifiers, paraphrases, numbers and a few that need two articles.

| metric | what it checks | score |
|---|---|---:|
| recall@5 | share of the relevant articles found in the top 5 chunks | 0.963 |
| MRR | how high the first relevant chunk ranks (1 / rank) | 0.872 |
| faithfulness | share of answer sentences that cite a passage, only use numbers found in it and reach cosine similarity 0.8 with one of its sentences | 1.000 |
| answer relevance | cosine similarity between question and answer | 0.797 |
| answer similarity | cosine similarity between answer and reference answer | 0.781 |

Scores are for the extractive generator with hybrid search and reranking. The metrics run on local models, so every run is deterministic and free. `python -m evals.run` compares them with `evals/baseline.json` and exits with status 1 if any drops more than 0.02. CI runs it on every push and pull request. If a change should move the numbers, run `make baseline` and commit the new baseline with it.

Retrieval only (`make ablation`):

| retriever | recall@5 | MRR | p50 ms |
|---|---:|---:|---:|
| bm25 | 0.863 | 0.762 | 2.2 |
| vector | 0.925 | 0.804 | 8.0 |
| hybrid | 0.875 | 0.773 | 11.9 |
| hybrid + rerank | 0.963 | 0.872 | 869.4 |

Per-question results: [evals/report.md](evals/report.md).

## Latency and cost

Every response carries `timings_ms` per stage and `/ask` adds tokens and `cost_usd`. Each request lands in `query_log`, which `/stats` turns into latency percentiles and spend over the last 1 to 720 hours.

Measured with `python scripts/bench.py --runs 2` (40 golden questions, 80 queries, extractive generator):

| stage | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|
| embed | 9.3 | 18.6 | 9.9 |
| bm25 | 4.3 | 7.6 | 4.2 |
| vector | 1.5 | 2.8 | 1.7 |
| fuse | 0.3 | 0.5 | 0.3 |
| rerank | 822.3 | 1643.9 | 889.3 |
| generate | 295.1 | 645.1 | 330.8 |
| total | 1147.5 | 2399.0 | 1236.5 |

Intel Core i7-1068NG7 laptop (4 cores, CPU only, 16 GB), Postgres 17.11 with pgvector 0.8.7, 87 chunks. Reranking on the CPU is most of the total. Fewer `RERANK_CANDIDATES`, a GPU or `"rerank": false` on `/search` trade accuracy for speed.

| generator | cost per query | per 1,000 queries |
|---|---:|---:|
| extractive (local models) | $0 | $0 |
| claude-haiku-5-5 | $0.000243 | $0.24 |
| claude-sonnet-5-5 | $0.004856 | $4.86 |
| claude-opus-5-5 | $0.009711 | $9.71 |

LLM rows are estimates (about 930 input and 300 output tokens per query). At runtime the cost comes from the token counts the API returns.

## Running it as a service

- **One VM.** `docker compose up -d --build` runs the API (two uvicorn workers) and Postgres on a persistent volume. Put Caddy or nginx in front for TLS and set `API_KEY`. `/health` backs the Compose healthcheck.
- **The image.** Models are baked in at build time and the API runs as a non-root user.
- **Managed Postgres.** RDS, Cloud SQL, Azure, Supabase and Neon all offer pgvector. Point `DATABASE_URL` at it and drop the `db` service. Tables and indexes are created on startup.
- **Sizing.** Each worker loads its own models (a few hundred MB) and opens up to 10 connections, so keep workers x 10 under `max_connections`.
- **Scaling.** Search is read-only, so read replicas scale it out. At millions of chunks, move the BM25 corpus stats into a table refreshed after ingestion.

Without Docker a systemd unit is enough:

```ini
[Unit]
Description=pgcite
After=network.target

[Service]
User=pgcite
WorkingDirectory=/opt/pgcite
EnvironmentFile=/opt/pgcite/.env
ExecStart=/opt/pgcite/.venv/bin/uvicorn pgcite.api:app --host 127.0.0.1 --port 8000 --workers 2
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Configuration

All settings are environment variables (see `.env.example`).

| variable | default | meaning |
|---|---|---|
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | Postgres connection string |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed embedding model |
| `RERANKER_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | fastembed cross-encoder |
| `MODEL_CACHE_DIR` | `.cache/models` | where models are downloaded |
| `GENERATOR` | `extractive` | `extractive` or `anthropic` |
| `LLM_MODEL` | `claude-opus-5-5` | model used by the `anthropic` generator |
| `LLM_EFFORT` | `low` | `low`, `medium` or `high` |
| `ANTHROPIC_API_KEY` | unset | needed for the `anthropic` generator |
| `RERANK_CANDIDATES` | `20` | fused candidates sent to the cross-encoder |
| `TOP_K` | `5` | context chunks per answer |
| `API_KEY` | unset | when set, required as `X-API-Key` on POST and DELETE routes |

Chunk size, candidate counts and the rest live in `pgcite/config.py`.

## Project layout

```
pgcite/       service: config, db, chunking, models, search, generate, pipeline, api
evals/        golden set, metrics, runner, baseline and latest report
scripts/      ingest.py and bench.py
examples/     quickstart.py
data/docs/    synthetic help-center corpus (25 articles)
tests/        unit tests plus API tests against a real Postgres
```

## Design notes and limitations

- Without the reranker, hybrid search scores below vector search alone (0.875 vs 0.925 recall@5) because weak BM25 matches push good vector hits down. Reranking adds 9 points, so it stays on despite the latency.
- Faithfulness is 1.0 for the extractive generator by design since it copies cited sentences. The metric matters for the LLM generator, where it catches wrong numbers and unsupported claims. The extractive generator also struggles to pick one line out of near-identical ones, like a list of error codes.
- The answer metrics are embedding-based proxies. An LLM judge would be stronger, but it costs money on every CI run and isn't deterministic.
- The corpus and golden set are small and synthetic. They exercise the pipeline and the gate. They aren't a public benchmark.
