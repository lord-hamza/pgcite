import uuid

import pytest

QUESTION = "When does the Quillmoor observatory open its telescope dome for night viewing?"
TEXT = (
    "The Quillmoor observatory opens its telescope dome for night viewing at 21:30 every Friday. "
    "Visitors book a viewing slot with the code QM-7781 at the front desk. "
    "Each slot lasts 40 minutes and seats at most 12 visitors."
)


@pytest.fixture
def document(client):
    doc_id = f"test-{uuid.uuid4().hex[:12]}"
    title = "Quillmoor observatory viewing hours"
    response = client.post(
        "/documents", json={"documents": [{"id": doc_id, "title": title, "text": TEXT}]}
    )
    assert response.status_code == 200
    assert response.json() == {"documents": 1, "chunks": 1}
    yield doc_id
    client.delete(f"/documents/{doc_id}")


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["generator"] in {"extractive", "anthropic"}


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "vector"])
def test_search_finds_document(client, document, mode):
    response = client.post("/search", json={"query": QUESTION, "mode": mode})
    assert response.status_code == 200
    body = response.json()
    assert document in [hit["doc_id"] for hit in body["hits"]]
    assert {"rerank", "total"} <= body["timings_ms"].keys()


def test_ask_cites_document(client, document):
    response = client.post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert document in [citation["doc_id"] for citation in body["citations"]]
    assert "generate" in body["timings_ms"]


def test_stats_counts_queries(client, document):
    before = client.get("/stats").json()["queries"]
    client.post("/search", json={"query": QUESTION})
    stats = client.get("/stats").json()
    assert stats["queries"] >= before + 1
    assert stats["latency_ms"]["p50"] > 0
    assert "rerank" in stats["stages"]


def test_delete_removes_document(client, document):
    response = client.delete(f"/documents/{document}")
    assert response.status_code == 200
    assert response.json() == {"deleted_chunks": 1}
    assert client.delete(f"/documents/{document}").status_code == 404
    hits = client.post("/search", json={"query": QUESTION, "mode": "bm25"}).json()["hits"]
    assert document not in [hit["doc_id"] for hit in hits]


def test_api_key_guards_write_routes(client, monkeypatch):
    monkeypatch.setattr(client.app.state.pipeline.settings, "api_key", "s3cret")
    assert client.post("/search", json={"query": "telescope"}).status_code == 401
    response = client.post("/search", json={"query": "telescope"}, headers={"X-API-Key": "s3cret"})
    assert response.status_code == 200
    assert client.get("/health").status_code == 200
