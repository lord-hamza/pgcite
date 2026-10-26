import pytest

from pgcite.search import Hit, rrf_fuse


def hit(chunk_id: int, score: float = 0.0) -> Hit:
    return Hit(chunk_id, f"doc-{chunk_id}", f"Title {chunk_id}", f"Content {chunk_id}", score)


def test_rrf_sums_reciprocal_ranks():
    fused = rrf_fuse([[hit(1), hit(2)], [hit(2), hit(3)]], k=60)
    scores = {h.chunk_id: h.score for h in fused}
    assert scores[1] == pytest.approx(1 / 61)
    assert scores[2] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[3] == pytest.approx(1 / 62)
    assert [h.chunk_id for h in fused] == [2, 1, 3]


def test_rrf_dedupes_by_chunk_id():
    fused = rrf_fuse([[hit(1, 9.0), hit(2)], [hit(1, 0.5)]])
    assert [h.chunk_id for h in fused] == [1, 2]
    assert fused[0].doc_id == "doc-1"


def test_rrf_ties_keep_first_seen_order():
    fused = rrf_fuse([[hit(5), hit(6)], [hit(7), hit(8)]])
    assert [h.chunk_id for h in fused] == [5, 7, 6, 8]


def test_rrf_returns_copies():
    original = hit(1, score=9.0)
    fused = rrf_fuse([[original]])
    assert original.score == 9.0
    assert fused[0] is not original


def test_rrf_handles_empty_rankings():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []
