import re

import numpy as np

from pgcite.chunking import split_sentences
from pgcite.generate import CITATION, NO_ANSWER, normalize_citations, strip_citations
from pgcite.models import Embedder

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Calibrated with bge-small on the golden set: the lowest value that rejects every unrelated
# sentence appended to a real answer (max 0.79) and keeps 82% of paraphrased reference sentences.
FAITHFULNESS_THRESHOLD = 0.8


def recall_at_k(retrieved_docs: list[str], relevant_docs: list[str], k: int) -> float:
    relevant = set(relevant_docs)
    return len(relevant & set(retrieved_docs[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved_docs: list[str], relevant_docs: list[str]) -> float:
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        if doc_id in relevant_docs:
            return 1 / rank
    return 0.0


def _numbers(text: str) -> set[str]:
    return {number.replace(",", "") for number in NUMBER.findall(text)}


def faithfulness(
    answer: str,
    contexts: dict[int, str],
    embedder: Embedder,
    threshold: float = FAITHFULNESS_THRESHOLD,
) -> float:
    """Fraction of answer sentences backed by the passages they cite.

    A sentence counts as supported when it cites a valid passage, every number in it appears in
    the cited text, and it is semantically close to at least one sentence of that text.
    """
    if strip_citations(answer) == NO_ANSWER:
        return 1.0
    claims = []
    for sentence in split_sentences(normalize_citations(answer)):
        refs = {int(ref) for ref in CITATION.findall(sentence)} & contexts.keys()
        claims.append((strip_citations(sentence), refs))
    cited = sorted(set().union(*(refs for _, refs in claims)))
    if not cited:
        return 0.0

    evidence = [(ref, sentence) for ref in cited for sentence in split_sentences(contexts[ref])]
    vectors = embedder.embed_passages([claim for claim, _ in claims] + [s for _, s in evidence])
    claim_vectors, evidence_vectors = vectors[: len(claims)], vectors[len(claims) :]
    evidence_refs = np.array([ref for ref, _ in evidence])

    supported = 0
    for (claim, refs), vector in zip(claims, claim_vectors, strict=True):
        cited_text = " ".join(contexts[ref] for ref in refs)
        if not refs or not _numbers(claim) <= _numbers(cited_text):
            continue
        mask = np.isin(evidence_refs, list(refs))
        if mask.any() and float((evidence_vectors[mask] @ vector).max()) >= threshold:
            supported += 1
    return supported / len(claims)


def _cosine(embedder: Embedder, first: str, second: str) -> float:
    a, b = embedder.embed_passages([first, second])
    return float(a @ b)


def answer_relevance(question: str, answer: str, embedder: Embedder) -> float:
    return _cosine(embedder, question, strip_citations(answer))


def answer_similarity(reference: str, answer: str, embedder: Embedder) -> float:
    return _cosine(embedder, reference, strip_citations(answer))
