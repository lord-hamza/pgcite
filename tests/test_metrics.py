import pytest

from evals.metrics import (
    answer_relevance,
    answer_similarity,
    faithfulness,
    recall_at_k,
    reciprocal_rank,
)
from pgcite.generate import NO_ANSWER, strip_citations

CONTEXTS = {
    1: "The Team plan costs 12 dollars per seat per month. Annual billing saves two months.",
    2: "Audit logs are retained for 365 days on the Enterprise plan.",
}
PRICE = "The Team plan costs 12 dollars per seat per month [1]."
AUDIT = "Audit logs are retained for 365 days on the Enterprise plan [2]."


def test_recall_at_k_is_doc_level():
    retrieved = ["a", "b", "a", "c"]
    assert recall_at_k(retrieved, ["a", "c"], k=2) == 0.5
    assert recall_at_k(retrieved, ["a", "c"], k=4) == 1.0
    assert recall_at_k([], ["a"], k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], ["a", "b"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_supported_answer_is_fully_faithful(embedder):
    assert faithfulness(f"{PRICE} {AUDIT}", CONTEXTS, embedder) == 1.0


def test_citation_after_punctuation_still_counts(embedder):
    answer = "Audit logs are retained for 365 days on the Enterprise plan. [2]"
    assert faithfulness(answer, CONTEXTS, embedder) == 1.0


def test_wrong_number_is_unsupported(embedder):
    answer = "The Team plan costs 15 dollars per seat per month [1]."
    assert faithfulness(answer, CONTEXTS, embedder) == 0.0


def test_uncited_sentence_is_unsupported(embedder):
    answer = f"{PRICE} Audit logs are retained for 365 days on the Enterprise plan."
    assert faithfulness(answer, CONTEXTS, embedder) == 0.5


def test_unrelated_claim_is_unsupported(embedder):
    answer = "Support replies within four hours on weekends [1]."
    assert faithfulness(answer, CONTEXTS, embedder) == 0.0


def test_appended_unrelated_sentence_is_unsupported(embedder):
    answer = f"{PRICE} Guests only see the pages that are shared with them [1]."
    assert faithfulness(answer, CONTEXTS, embedder) == 0.5


def test_answer_with_citations_removed_is_unsupported(embedder):
    answer = strip_citations(f"{PRICE} {AUDIT}")
    assert faithfulness(answer, CONTEXTS, embedder) == 0.0


def test_citation_to_missing_passage_is_unsupported(embedder):
    answer = "The Team plan costs 12 dollars per seat per month [7]."
    assert faithfulness(answer, CONTEXTS, embedder) == 0.0


def test_no_answer_counts_as_faithful(embedder):
    assert faithfulness(NO_ANSWER, CONTEXTS, embedder) == 1.0


def test_relevance_and_similarity(embedder):
    question = "How long are audit logs retained?"
    assert answer_relevance(question, AUDIT, embedder) > answer_relevance(question, PRICE, embedder)
    reference = "Audit logs are retained for 365 days on the Enterprise plan."
    assert answer_similarity(reference, AUDIT, embedder) == pytest.approx(1.0)
    assert answer_similarity(reference, PRICE, embedder) < 0.5
